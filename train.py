import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import rssm_common
from container import BioPong  # Assuming BioPong is in a file named BioPong.py
from collections import deque
import os
from tensorflow.keras import layers, models
from tqdm import tqdm
import random
from utils import *

# Set up TensorFlow environment and other configurations
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorboardX import SummaryWriter
writer = SummaryWriter('logs/biopong_learn_fresh')

# Initialize the RSSM with MLP components instead of CNN
config_rssm = {'ensemble': 2, 'hidden': 600, 'deter': 600, 'stoch': 32, 'discrete': 32, 'act': 'elu', 'norm': 'none', 'std_act': 'sigmoid2', 'min_std': 0.1}
rssm = rssm_common.EnsembleRSSM(**config_rssm)

config_decoder = {'mlp_keys': 'image', 'cnn_keys': '$^', 'act': 'elu', 'norm': 'none', 'cnn_depth': 48, 'cnn_kernels': [5, 5, 6, 6], 'mlp_layers': [400, 400, 400, 400]}
shapes =  {'image': (72,), 'reward': (), 'is_first': (), 'is_last': (), 'is_terminal': ()}
decoder = rssm_common.Decoder(shapes, **config_decoder)

config_encoder = {'mlp_keys': 'image', 'cnn_keys': '$^', 'act': 'elu', 'norm': 'none', 'cnn_depth': 48, 'cnn_kernels': [4, 4, 4, 4], 'mlp_layers': [400,400,400,400]}
encoder = rssm_common.Encoder(shapes, **config_encoder)

modules = [decoder, rssm, encoder]
config_model_opt =  {'opt': 'adam', 'lr': 1e-4, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6}
rssm_optimizer = rssm_common.Optimizer('rssm', **config_model_opt)

checkpoint_dir = './checkpoints_dir_vae'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(rssm_optimizer=rssm_optimizer,
                                 decoder=decoder,
                                 rssm=rssm,
                                 encoder=encoder)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restores the latest checkpoint using the manager.
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


class PlannerNet():
    def __init__(self, rssm, decoder):
        super(PlannerNet, self).__init__()
        self.rssm = rssm
        self.decoder = decoder

    def call(self, prior, sdr_24bit_basis, action_tensors, mode):
        
        if mode == 0:
            future_states = imagine_actions(prior, action_tensors, self.decoder, self.rssm, "imagine")

            tensor_future_array = batch_process_future_states(future_states, sdr_24bit_basis, temperature=0.01)
            
            differences = tf.reduce_sum(tf.abs((tensor_future_array[:,:,0]+(1/12)) - tensor_future_array[:,:,2]),axis=1)

            min_difference_index = tf.argmin(differences, axis=0)

            selected_action = tf.gather(action_tensors, min_difference_index)

            sel_future_array1 = tensor_future_array[min_difference_index]-0.5
            sel_future_array2 = tensor_future_array[min_difference_index]-0.5
        

        elif mode == 1:
            future_states1 = imagine_actions(prior, action_tensors, self.decoder, self.rssm, "imagine_1")
            tensor_future_array1 = batch_process_future_states(future_states1, sdr_24bit_basis, temperature=0.01)
            future_states2 = imagine_actions(prior, action_tensors, self.decoder, self.rssm, "imagine_2")
            tensor_future_array2 = batch_process_future_states(future_states2, sdr_24bit_basis, temperature=0.01)

            differences = tf.reduce_sum(tf.abs(tensor_future_array1 - tensor_future_array2),axis=[1, 2])

            min_difference_index = tf.argmin(differences, axis=0)

            selected_action = tf.gather(action_tensors, min_difference_index)

            sel_future_array1 = tensor_future_array1[min_difference_index]-0.5
            sel_future_array2 = tensor_future_array2[min_difference_index]-0.5

        return differences, selected_action, sel_future_array1, sel_future_array2

def main():

    frame_list = deque(maxlen=10000) # collect sdr values
    value_list = deque(maxlen=10000) # collect vector values
    action_list = deque(maxlen=10000) # collect actions
    nll_list = deque(maxlen=10000) # collect matching measure
    pred_action_seq = []
    
    batch_size = 16
    time_steps = 100
    sequence_length = 100
    time_actions = 5
    action_onehot =[[1,0,0],[0,1,0],[0,0,1]]

    exploration_actions = create_exploration_actions(sequence_length)
    sdr_24bit_basis = create_sdr_basis(res_basis = 8)

    current_rally = 0
    train_rallies = 200

    network = PlannerNet(rssm, decoder)
    train_mode = False
    train_steps = 10
    save_checkpoint = False

    env = BioPong()
    state = env.reset()
    while current_rally < train_rallies:

        if len(pred_action_seq) > 0:
            action = pred_action_seq[0]
            action_i = np.argmax(action)
            pred_action_seq = pred_action_seq[1:]
            action_list.append(action)
        else:
            action_i = np.random.choice([0, 1, 2])
            action_list.append(action_onehot[action_i])

        next_state = env.step(action_i)
        sdr_state = np.array(array_to_sdr(next_state))

        frame_list.append(sdr_state)
        value_list.append(next_state)

        rally, score = env._get_score()
        if rally != current_rally:
            current_rally = rally
            print(f'rally: {current_rally}, score: {score}')
            writer.add_scalar('score', score, rally)

            if train_mode and len(frame_list) > 2*time_steps:
                frame_array = np.array(frame_list)
                value_array = np.array(value_list)
                action_array = np.array(action_list)
                nll_array = np.array(nll_list)

                for step in tqdm(range(train_steps), desc="Learning", unit="step"):

                    start_indices = sample_nll_indices(nll_array[:len(frame_array) - time_steps], batch_size//2)
                    start_indices = np.append(start_indices,np.random.randint(0, len(frame_array) - time_steps, batch_size//2))
                
                    batched_experience_list = [frame_array[idx:idx + time_steps] for idx in start_indices]
                    batched_label_list = [value_array[idx:idx + time_steps] for idx in start_indices]
                    act = [action_array[idx:idx + time_steps] for idx in start_indices]
                    x = tf.convert_to_tensor(batched_experience_list, dtype=tf.float32)
                    y = tf.convert_to_tensor(batched_label_list, dtype=tf.float32)
                    act = tf.convert_to_tensor(act, dtype=tf.float32)
                    
                    with tf.GradientTape(persistent=True) as tape:
                        model_loss, kl_loss, nll_loss = compute_loss(x, act, rssm, decoder, encoder)
                    rssm_optimizer(tape, model_loss, modules)

                if save_checkpoint and (current_rally > 0) and (current_rally % 3 == 0):
                    save_path = manager.save()
                    print("Saved checkpoint: {}".format(save_path))

        if len(frame_list) > time_actions:
            frame_array = np.array(frame_list)[-time_actions:]
            action_array = np.array(action_list)[-time_actions:]

            x = tf.convert_to_tensor([frame_array], dtype=tf.float32)
            act = tf.convert_to_tensor([action_array], dtype=tf.float32)

            if len(pred_action_seq) == 0:

                if train_mode and len(frame_list) > 2 * time_steps:

                    if rally > 1:

                        frame_array = np.array(frame_list)
                        action_array = np.array(action_list)

                        start_indices = find_similar_traj(np.array(value_list), sample = batch_size//2)
                        start_indices = np.append(start_indices,np.random.randint(0, len(frame_array) - time_steps, batch_size//2))
                    
                        batched_experience_list = [frame_array[idx:idx + time_steps] for idx in start_indices]
                        act_traj = [action_array[idx:idx + time_steps] for idx in start_indices]

                        x_traj = tf.convert_to_tensor(batched_experience_list, dtype=tf.float32)
                        act_traj = tf.convert_to_tensor(act_traj, dtype=tf.float32)
                        
                        with tf.GradientTape(persistent=True) as tape:
                            model_loss, kl_loss, nll_loss = compute_loss(x_traj, act_traj, rssm, decoder, encoder)
                        
                        rssm_optimizer(tape, model_loss, modules)
                    
                best_cost = None
                best_action_seq = None

                init_sequence_length =  len(exploration_actions)

                x = tf.tile(x, [init_sequence_length, 1, 1])
                act = tf.tile(act, [init_sequence_length, 1, 1])

                actions_shape = tf.shape(act)
                is_first = tf.zeros([actions_shape[0], actions_shape[1]], dtype=tf.bool)

                init_state, prior, nll_loss = imagine_init(x, act, is_first, decoder, encoder, rssm)
                nll_list.append(nll_loss)

                cost, action_seq, output1, output2 = network.call(prior, sdr_24bit_basis, exploration_actions, mode=0)
                for action in action_seq[:20]:
                    pred_action_seq.append(action)

                env.render_traj(next_state[0],next_state[1],next_state[2], output1)
                
            else:
                actions_shape = tf.shape(act)
                is_first = tf.zeros([actions_shape[0], actions_shape[1]], dtype=tf.bool)

                init_state, prior, nll_loss = imagine_init(x, act, is_first, decoder, encoder, rssm)
                nll_list.append(nll_loss)
        
                output_array = continous_sdr_to_array(init_state, sdr_24bit_basis)

                env.render2_traj(next_state[0],next_state[1],next_state[2], output_array[0], output_array[1], output_array[2])

        else:
            nll_list.append(0)
            
            
if __name__ == "__main__":
    main()
       