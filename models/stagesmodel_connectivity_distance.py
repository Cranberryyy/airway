import tensorflow as tf
import os
import time
import numpy as np
from .base import net_collection, optim_collection, loss_collection, HardCasePool 
from .utils import eval_metric, save_image, create_boundary, create_connectivity 
from utils import print_evaluation_log, save_nii, print_training_log

class StagesModel_Connectivity_Distance():
    def __init__(self, args, net_opt):
        self.args = args
        self.net_opt =  net_opt[args.net]
        self.loss_type = '_'.join([self.args.out_loss_1, self.args.out_loss_2, self.args.deep_loss_1, self.args.deep_loss_2])
        self.save_model_dir = os.path.join(args.model_dir, args.model, '%s_%s' %(args.net, self.loss_type))
        self.optim_func = optim_collection[args.optimizer]
        self.build_network()
        self.log_fid = self.create_log(args)
        self.hardcase_pool = HardCasePool(args.pool_size)
        

    def create_log(self, args):
        model_dir = self.save_model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        name = 'log-%s.txt' %(cur_time)
        path = os.path.join(model_dir, name)
        log_fid = open(path, 'w')
        return log_fid  
    
    def build_network(self):
        self.used_net_1 = net_collection[self.args.net](**self.net_opt)
        self.used_net_2 = net_collection[self.args.net](**self.net_opt)
        channel = 4 if self.args.load_coord else 1
        input = tf.keras.layers.Input(shape=(self.args.cropping_size[0], self.args.cropping_size[1], self.args.cropping_size[2], channel))
        gt = tf.keras.layers.Input(shape=(self.args.cropping_size[0], self.args.cropping_size[1], self.args.cropping_size[2], 1))
        connectivity = create_connectivity(gt, name='create_connectivity')
        self.create_connectivity = tf.keras.Model(inputs=gt, outputs=connectivity)
        distance = create_boundary(gt, name='create_boundary')
        self.create_boundary = tf.keras.Model(inputs=gt, outputs=distance)


        self.output_1, encode_feat = self.used_net_1(input, is_mid_feat_in=False, is_mid_feat_out=True, is_distance=True, name='net_1')
        self.output_2 = self.used_net_2(encode_feat, is_mid_feat_in=True, is_mid_feat_out=False, is_distance=True, name='net_2')
        self.train_net_1 = tf.keras.Model(inputs=input, outputs=self.output_1)
        self.train_net_2 = tf.keras.Model(inputs=input, outputs=self.output_2)
        self.train_net = tf.keras.Model(inputs=input, outputs=[self.output_1, self.output_2])

        self.train_net_1_trainable_variables = self.train_net_1.trainable_variables
        self.train_net_2_trainable_variables = [var for var in self.train_net_2.trainable_variables if 'net_2' in var.name]

    def resume(self, args):
        if args.continue_train and args.resume_dir is None:
            model_dir = self.save_model_dir
            model_path  = tf.train.latest_checkpoint(model_dir)
            self.train_net.load_weights(model_path)
            print('Loading model from {} successfully !'.format(model_path))
        
        if args.continue_train and args.resume_dir is not None:
            model_path  = tf.train.latest_checkpoint(args.resume_dir)
            self.train_net.load_weights(model_path)
            print('Loading model from {} successfully !'.format(args.resume_dir))
            
        if args.model_stage == 2 and args.continue_train and args.reset_head2:
            for _layer in self.train_net_2.layers:
                if 'net_2' in _layer.name:
                    new_layer_name = _layer.name.replace('net_2', 'net_1')
                    weights = self.train_net_1.get_layer(new_layer_name).get_weights()
                    _layer.set_weights(weights)
            print('Resetting head 2 successfully !')


    def padding(self, input, shape):
        cur_shape = input.shape[1:4]
        pad_num = ((0, shape[0]-cur_shape[0]), (0, shape[1]-cur_shape[1]), (0, shape[2]-cur_shape[2]))
        return tf.keras.layers.ZeroPadding3D(padding=pad_num)(input)
    
    def save_model(self, args, epoch):
        model_dir = self.save_model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model-{}'.format(epoch))
        self.train_net.save_weights(model_path)
        print('Saved model to {} successfully !'.format(model_path))

    def loss_metric(self, gt, net_out, gt_connectivity, gt_distance, head=1, beta=0.7, gama=1):
        out_loss = self.args.out_loss_1 if head == 1 else self.args.out_loss_2
        deep_loss = self.args.deep_loss_1 if head == 1 else self.args.deep_loss_2
        is_recall = True if head == 1 and self.args.model_stage==2 else False
        final_net_out_seg = net_out[0]
        final_net_out_connect = net_out[1]
        final_net_out_distance = net_out[2]
        deep_net_out = net_out[3:]
        gt_onehot = tf.squeeze(tf.one_hot(tf.cast(gt, tf.int32), depth=final_net_out_seg.shape[-1]),-2)
        final_loss = loss_collection['tversky'](gt_onehot, final_net_out_seg, beta=beta)+ loss_collection['wce'](gt_onehot, final_net_out_seg, gama=gama, is_recall=is_recall)
        connec_loss = loss_collection['dice'](gt_connectivity, final_net_out_connect)
        distance_loss = loss_collection['boundary'](gt_distance, final_net_out_distance)
        mid_loss = 0
        for out in deep_net_out:
            if out.shape[-1] != gt.shape[-1]:
                mid_loss += loss_collection[deep_loss](gt_onehot, out)
            else:
                mid_loss += loss_collection[deep_loss](gt, out)
        loss = self.args.w_final_loss * final_loss + self.args.w_connec_loss * connec_loss \
              + self.args.w_mid_loss * mid_loss + self.args.w_distance_loss * distance_loss
        return loss #, [final_loss, connec_loss, mid_loss, distance_loss]

    @tf.function
    def train_step(self, input, gt, gt_s=None, stage=1):
        gt_distance = self.create_boundary(gt)
        gt_connectivity = self.create_connectivity(gt)
        if stage == 1:
            """train general example"""
            with tf.GradientTape() as tape:
                net_out = self.train_net_1(input)
                loss = self.loss_metric(gt, net_out, gt_connectivity, gt_distance, head=1, gama=0.5)
            gradients = tape.gradient(loss, self.train_net_1_trainable_variables)
            self.optim_head1.apply_gradients(zip(gradients, self.train_net_1_trainable_variables))
        else:
            net_out = self.train_net_2(input)
            pred = tf.cast(tf.expand_dims(tf.argmax(net_out[0], axis=-1), axis=-1), tf.float32)
            index = tf.reduce_sum(pred * gt_s, axis=[1,2,3,4]) / tf.reduce_sum(gt_s, axis=[1,2,3,4]) < 0.93
            if tf.reduce_sum(tf.cast(index, tf.int32)) > 0:
                """train header 1"""
                with tf.GradientTape() as tape:  ####hard example
                    net_out = self.train_net_1(input)
                    net_out = [tf.boolean_mask(temp, index, axis=0) for temp in net_out]
                    loss_1 = self.loss_metric(tf.boolean_mask(gt, index, axis=0), net_out,
                                              tf.boolean_mask(gt_connectivity, index, axis=0),
                                                tf.boolean_mask(gt_distance, index, axis=0),
                                                  head=1, gama=0.1, beta=0.8)
                gradients = tape.gradient(loss_1, self.train_net_1_trainable_variables)
                self.optim_head1.apply_gradients(zip(gradients, self.train_net_1_trainable_variables))
            else:
                loss_1 = 0.0

            """train header 2"""
            with tf.GradientTape() as tape:
                net_out = self.train_net_2(input)
                loss_2 = self.loss_metric(gt, net_out, gt_connectivity, gt_distance, head=2, gama=10, beta=0.5)
            gradients = tape.gradient(loss_2, self.train_net_2_trainable_variables)
            self.optim_head2.apply_gradients(zip(gradients, self.train_net_2_trainable_variables))
            loss = [loss_1, loss_2]
        return loss
    
    def train(self, args, train_loader, info):
        args.num_train_steps = int(args.epoch * info['training']['samples'] / args.batch_size);    
        learning_rate_head1=args.learning_rate_head1
        learning_rate_head2=args.learning_rate_head2
        samples_in_epoch = info['training']['samples']
        steps_in_epoch = int(samples_in_epoch / args.batch_size)
        for epoch in range(args.epoch):
            if epoch%10 == 0:
                if epoch != 0:
                    learning_rate_head1 = max(learning_rate_head1 * 0.5, 1e-6)
                    learning_rate_head2 = max(learning_rate_head2 * 0.5, 1e-6)
                self.optim_head1 = self.optim_func(learning_rate=learning_rate_head1)
                self.optim_head2 = self.optim_func(learning_rate=learning_rate_head2)
            self.train_epoch(train_loader, args, epoch, steps_in_epoch)
                
    def train_val(self, args, train_loader, val_loader, info):
        args.num_train_steps = int(args.epoch * info['training']['samples'] / args.batch_size);    
        learning_rate_head1=args.learning_rate_head1
        learning_rate_head2=args.learning_rate_head2
        samples_in_epoch = info['training']['samples']
        steps_in_epoch = int(samples_in_epoch / args.batch_size)
        val_total_case = info['validating']['case']
        self.optim_head1 = self.optim_func(learning_rate=learning_rate_head1)
        self.optim_head2 = self.optim_func(learning_rate=learning_rate_head2)
        for epoch in range(1, args.epoch+1, 1):
            if epoch%10 == 0:
                learning_rate_head1 = max(learning_rate_head1 * 0.5, 1e-6)
                learning_rate_head2 = max(learning_rate_head2 * 0.5, 1e-6)
                self.optim_head1 = self.optim_func(learning_rate=learning_rate_head1)
                self.optim_head2 = self.optim_func(learning_rate=learning_rate_head2)
            if epoch % args.val_epoch == 0 and epoch != 1:
                self.val_epoch(val_loader, args, val_total_case)
            self.train_epoch(train_loader, args, epoch, steps_in_epoch)

    def test(self, args, test_loader, info):
        test_total_case = info['testing']['case']
        self.val_epoch(test_loader, args, test_total_case)

    def val(self, args, val_loader, info):
        val_total_case = info['val']['case']
        self.val_epoch(val_loader, args, val_total_case)

    def train_epoch(self, train_loader, args, epoch, steps_in_epoch):
        step = 0 
        for train_data_batch in train_loader: 
            """ Training """
            image, label, dist, _ = train_data_batch
            label = tf.cast(label>0, tf.float32)
            patches_gt, patches_gt_s = label[..., 0:1], label[..., 1:2]
            if self.args.load_coord:
                if image.shape[-2] != dist.shape[-2]:
                    continue
                patches_in = tf.concat([image, dist], axis=-1)
            try:
                loss = self.train_step(patches_in, patches_gt, patches_gt_s, args.model_stage)
            except:
                print('error')
            step += 1
            print_training_log(args, epoch, step, steps_in_epoch, loss, self.log_fid)
            if step >= steps_in_epoch:
                break
            if step % (steps_in_epoch // 2) == 0:
                 self.save_model(args, epoch)
        self.save_model(args, epoch)

    def val_epoch(self, val_loader, args, val_total_case):
        val_case = 0; result = 0; result_2 = 0; 
        if args.mode == 'Train_Val':
            save_file = 'train'
        else:
            save_file = 'test'
        for val_data_batch in val_loader:
            flag = True
            if len(val_data_batch) == 6:
                image, dist, affine, box, ori_shape, name  = val_data_batch
                flag = False
            else:
                image, parse, dist, skel, affine, box, _, _, name  = val_data_batch
            if not isinstance(name, str):
                name = str(name.numpy(), 'UTF-8')
            if self.args.load_coord:
                patches_in = tf.concat([image, dist], axis=-1)
            pred_1, pred_2, prob_map1, prob_map2 = self.val_test_step(args, patches_in)
            if args.save_val:
                save_path = os.path.join(self.save_model_dir, save_file)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if flag:
                    save_image(pred_2, affine, name, save_path)
                    save_image(pred_1, affine, name.replace('.nii.gz', '_head1.nii.gz'), save_path)
                    res = eval_metric(pred_1, parse, skel)
                    res_2 = eval_metric(pred_2, parse, skel)
                    result += res; result_2 += res_2
                    val_case += 1
                    print_evaluation_log('Head 1, VAL CASE %02d ' %(val_case), res, self.log_fid)
                    print_evaluation_log('Head 2, VAL CASE %02d ' %(val_case), res_2, self.log_fid)
                    np.save(os.path.join(save_path, name.replace('.nii.gz', '_prob1.npy')), prob_map1)
                else:
                    print("NO GT for currtent case: %s" %(name))
                    save_image(pred_2, affine, name.replace('.nii.gz', '_crop.nii.gz'), save_path)
                    save_image(pred_1, affine, name.replace('.nii.gz', '_crop_head1.nii.gz'), save_path)
                    save_image(prob_map2, affine, name.replace('.nii.gz', '_prob2.nii.gz'), save_path)
                    save_image(prob_map1, affine, name.replace('.nii.gz', '_prob1.nii.gz'), save_path)
                    val_case += 1
            if val_case == val_total_case:
                break
        if flag:
            print_evaluation_log('Head 1, VAL CASE AVG', result/val_case, self.log_fid)
            print_evaluation_log('Head 2, VAL CASE AVG', result_2/val_case, self.log_fid)


    def val_test_step(self, args, input):
        step = args.slide_window_step; shape = args.cropping_size; cnt = 0
        oc = self.train_net_1.outputs[0].shape[-1]
        step = step if (np.array(step)>4).any() else shape // step
        indexes = []; index_map = np.zeros(input.shape[0:3]+[1], dtype=np.int32)
        prob_map = np.zeros(input.shape[0:3]+[oc], dtype=np.float32)
        prob_map_2 = np.zeros(input.shape[0:3]+[oc], dtype=np.float32)
        for i in range(0, input.shape[0], step[0]):
            for j in range(0, input.shape[1], step[1]):
                for k in range(0, input.shape[2], step[2]):
                    input_temp = tf.expand_dims(input[i:i+shape[0], j:j+shape[1], k:k+shape[2]], axis=0)
                    cur_input_shape = input_temp.shape[1:4]; index = [i, j, k]
                    input_batch = input_temp if input_temp.shape[1:4] == shape else self.padding(input_temp, shape)
                    if cnt == 0:
                        input_batches = input_batch; input_shapes = [cur_input_shape]; indexes = [index]; cnt += 1
                    elif cnt <= args.batch_size*4:
                        input_batches = tf.concat([input_batches, input_batch], axis=0); input_shapes.append(cur_input_shape); indexes.append(index); cnt += 1
                    if cnt == args.batch_size*2:
                        output, output_2 = self.train_net(input_batches, training=False)
                        if len(output) > 1:
                            output = output[0]
                            output_2 = output_2[0]
                        for idx in range(cnt):
                            cur_input_shape = input_shapes[idx]; ii, jj, kk = indexes[idx]
                            prob_map[ii:ii+shape[0], jj:jj+shape[1], kk:kk+shape[2]] += \
                                        output[idx, 0:cur_input_shape[0], 0:cur_input_shape[1], 0:cur_input_shape[2]].numpy()
                            prob_map_2[ii:ii+shape[0], jj:jj+shape[1], kk:kk+shape[2]] += \
                                        output_2[idx, 0:cur_input_shape[0], 0:cur_input_shape[1], 0:cur_input_shape[2]].numpy()
                            index_map[ii:ii+shape[0], jj:jj+shape[1], kk:kk+shape[2]] += 1; cnt = 0
        if cnt > 0:
            output, output_2 = self.train_net(input_batches, training=False)
            if len(output) > 1:
                output = output[0]
                output_2 = output_2[0]
            for idx in range(cnt):
                cur_input_shape = input_shapes[idx]; ii, jj, kk = indexes[idx]
                prob_map[ii:ii+shape[0], jj:jj+shape[1], kk:kk+shape[2]] += \
                            output[idx, 0:cur_input_shape[0], 0:cur_input_shape[1], 0:cur_input_shape[2]].numpy()
                prob_map_2[ii:ii+shape[0], jj:jj+shape[1], kk:kk+shape[2]] += \
                            output_2[idx, 0:cur_input_shape[0], 0:cur_input_shape[1], 0:cur_input_shape[2]].numpy()
                index_map[ii:ii+shape[0], jj:jj+shape[1], kk:kk+shape[2]] += 1
        if prob_map.shape[-1] == 1:
            prob_map = tf.divide(prob_map, tf.cast(index_map, tf.float32))
            prob_map_2 = tf.divide(prob_map_2, tf.cast(index_map, tf.float32))
            pred = tf.squeeze(tf.cast(prob_map > 0.5, tf.int32), axis=-1)
            pred = tf.expand_dims(tf.cast(pred, tf.float32), axis=0)
            pred_2 = tf.squeeze(tf.cast(prob_map_2 > 0.5, tf.int32), axis=-1)
            pred_2 = tf.expand_dims(tf.cast(pred_2, tf.float32), axis=0)
        else:
            pred = tf.argmax(prob_map, axis=-1)
            pred = tf.expand_dims(tf.cast(pred, tf.float32), axis=0)
            pred_2 = tf.argmax(prob_map_2, axis=-1)
            pred_2 = tf.expand_dims(tf.cast(pred_2, tf.float32), axis=0)
            prob_map = prob_map[:,:,:,1]
            prob_map_2 = prob_map_2[:,:,:,1]
        return pred, pred_2, prob_map, prob_map_2

