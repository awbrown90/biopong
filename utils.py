import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from container import BioPong  # Assuming BioPong is in a file named BioPong.py
from collections import deque
import os
from tensorflow.keras import layers, models
from tqdm import tqdm
import random

def value_to_24bit_sdr(value):
    # Normalize the value to [0, 1]
    normalized_value = (value + 0.5) 

    # Stage 1: Determine the primary segment (0-7)
    primary_segment = int(normalized_value * 8)
    sdr = [1 if i == primary_segment else 0 for i in range(8)]

    # Stage 2: Subdivision within the primary segment
    within_segment = (normalized_value * 8) % 1  # Get fractional part
    sub_segment = int(within_segment * 8)
    sdr.extend([1 if i == sub_segment else 0 for i in range(8)])

    # Stage 3: Further subdivision
    within_sub_segment = (within_segment * 8) % 1  # Get fractional part
    sub_sub_segment = int(within_sub_segment * 8)
    sdr.extend([1 if i == sub_sub_segment else 0 for i in range(8)])

    return sdr

def array_to_sdr(array):
    sdr_output = []
    for value in array:
        sdr_output.extend(value_to_24bit_sdr(value))
    return sdr_output

def decode_24bit_sdr_to_value(sdr_24bit):
    # Decode primary segment
    primary_segment = sdr_24bit[:8].index(1) / 8

    # Decode subdivision
    sub_segment = sdr_24bit[8:16].index(1) / 64  # 1/8 for the segment size and 1/8 for position within segment

    # Decode further subdivision
    sub_sub_segment = sdr_24bit[16:24].index(1) / 512  # 1/8 for the segment size and 1/64 for position within subdivision

    # Combine and denormalize
    value = (primary_segment + sub_segment + sub_sub_segment) - 0.5
    return value

def sdr_to_array(sdr_72bit):
    return [decode_24bit_sdr_to_value(sdr_72bit[i:i+24]) for i in range(0, len(sdr_72bit), 24)]

def process_list_of_arrays(input_list):
    output_nested_list = []
    for sublist in input_list:
        output_sublist = []
        for array in sublist:
            sdr_representation = array_to_sdr(array)  # Use the previously defined array_to_sdr function
            output_sublist.append(sdr_representation)
        output_nested_list.append(output_sublist)
    return output_nested_list

def continuous_sdr_to_binary(sdr, segment_length=8):
    binary_sdr = []
    for i in range(0, len(sdr), segment_length):
        segment = sdr[i:i + segment_length]
        # Convert TensorFlow tensor to numpy array if needed
        if tf.is_tensor(segment):
            segment = segment.numpy()

        max_index = segment.argmax()
        binary_segment = [1 if j == max_index else 0 for j in range(segment_length)]
        binary_sdr.extend(binary_segment)
    return binary_sdr

def continous_sdr_to_array(sdr, sdr_24bit_basis, temperature=0.01):
    segments = tf.split(sdr, 3, axis=0)  # Split into three segments
    parts = [tf.split(segment, 3, axis=-1) for segment in segments]  # Split each segment into three parts

    # Apply softmax to each part and flatten
    flattened_parts = [tf.reshape(tf.nn.softmax(part / temperature), [-1]) for segment in parts for part in segment]

    # Concatenate all parts to form the final array
    final_softmax_array = tf.concat(flattened_parts, axis=0)

    # Multiply with sdr_24bit_basis and decode
    segment_length = 24
    predicted_output = [tf.reduce_sum(final_softmax_array[i*segment_length:(i+1)*segment_length] * sdr_24bit_basis) - 0.5 for i in range(len(final_softmax_array) // segment_length)]
    return predicted_output

def compute_loss(states, actions, rssm, decoder, encoder, training=True):
   
    # Embedding and Observing
    #print(states.shape)
    
    actions_shape = tf.shape(actions)
    is_first = tf.zeros([actions_shape[0], actions_shape[1]], dtype=tf.bool)

    #embed = encoder({'image':states}, training=training)
    embed = encoder({'image':states})
    post, prior = rssm.observe(embed, actions, is_first)
    
    # KL Loss
    kl = {'free': 0.0, 'forward': False, 'balance': 0.8, 'free_avg': True}
    kl_loss, kl_value = rssm.kl_loss(post, prior, **kl)
    
    # Get features and detach for decoder
    feat = rssm.get_feat(post)
    
    # Decoder NLL Loss
    dist = decoder(feat)['image']

    nll_loss = -tf.cast(dist.log_prob(states), tf.float32).mean()

    # Combine losses with weights from config
    weighted_kl_loss = 0.1 * kl_loss
    weighted_nll_loss = 1.0 * nll_loss
    total_loss = weighted_kl_loss + weighted_nll_loss
    
    return total_loss, kl_loss, nll_loss

def imagine_init(states, actions, is_first, decoder, encoder, rssm):
    embed = encoder({'image':states})
    #print("Embed shape after encoder:", embed.shape)
    #print(f'embed {embed}')
    start_idx = 5
    enc_states, _ = rssm.observe(
          embed[:,:start_idx], actions[:,:start_idx], is_first[:,:start_idx])
    init = {k: v[:, -1] for k, v in enc_states.items()}
    features = rssm.get_feat(enc_states)
    init_state = decoder(features)['image'].mode()

    dist = decoder(features)['image']
    nll_loss = -tf.cast(dist.log_prob(states), tf.float32).mean()

    return init_state[-1][-1], init, nll_loss

def imagine_init_1(states, actions, is_first, decoder, encoder, rssm):
    embed = encoder({'image':states})
    #print("Embed shape after encoder:", embed.shape)
    #print(f'embed {embed}')
    start_idx = 5
    enc_states, _ = rssm.observe_1(
          embed[:,:start_idx], actions[:,:start_idx], is_first[:,:start_idx])
    init = {k: v[:, -1] for k, v in enc_states.items()}
    features = rssm.get_feat(enc_states)
    init_state = decoder(features)['image'].mode()
    return init_state[-1][-1], init

def imagine_init_2(states, actions, is_first, decoder, encoder, rssm):
    embed = encoder({'image':states})
    #print("Embed shape after encoder:", embed.shape)
    #print(f'embed {embed}')
    start_idx = 5
    enc_states, _ = rssm.observe_2(
          embed[:,:start_idx], actions[:,:start_idx], is_first[:,:start_idx])
    init = {k: v[:, -1] for k, v in enc_states.items()}
    features = rssm.get_feat(enc_states)
    init_state = decoder(features)['image'].mode()
    return init_state[-1][-1], init

def imagine_action(init, action, decoder, rssm, imagine_function_name):
    # Use getattr to dynamically call the appropriate imagine function
    prior = getattr(rssm, imagine_function_name)(action, init)
    features = rssm.get_feat(prior)
    next_state = decoder(features)['image'].mode()
    prior = {k: v[:, -1] for k, v in prior.items()}
    return next_state[0], prior

def imagine_actions(init, action, decoder, rssm, imagine_function_name):
    # Use getattr to dynamically call the appropriate imagine function
    prior = getattr(rssm, imagine_function_name)(action, init)
    features = rssm.get_feat(prior)
    next_state = decoder(features)['image'].mode()
    return next_state

def imagine_features(init, action, decoder, rssm, imagine_function_name):
    # Use getattr to dynamically call the appropriate imagine function
    prior = getattr(rssm, imagine_function_name)(action, init)
    features = prior['deter']
    return features

def batch_process_future_states(future_states, sdr_24bit_basis, temperature):
    # Reshape future_states to merge batch and time dimensions
    reshaped_states = tf.reshape(future_states, [-1, future_states.shape[-1]])

    # Split each 72-dimensional vector into three 24-dimensional vectors
    split_states = tf.split(reshaped_states, num_or_size_splits=3, axis=1)

    # Apply softmax with temperature to each 8-dimensional part of the 24-dimensional vectors
    transformed_vectors = []
    for segment in split_states:
        # Split each 24-dimensional segment into three 8-dimensional vectors
        sub_segments = tf.split(segment, num_or_size_splits=3, axis=1)

        # Apply softmax to each 8-dimensional vector and concatenate back to 24-dimensional vector
        softmaxed_segments = [tf.nn.softmax(sub_segment / temperature, axis=1) for sub_segment in sub_segments]
        softmaxed_segment = tf.concat(softmaxed_segments, axis=1)

        # Element-wise multiplication with 1D sdr_24bit_basis
        multiplied_segment = softmaxed_segment * sdr_24bit_basis
        summed_segment = tf.reduce_sum(multiplied_segment, axis=1)

        # Reshape each scalar to a vector for concatenation
        reshaped_segment = tf.reshape(summed_segment, [-1, 1])
        transformed_vectors.append(reshaped_segment)

    # Concatenate the transformed vectors to form a 3-dimensional vector
    final_output = tf.concat(transformed_vectors, axis=1)

    # Reshape back to the original shape (21, 10, 3)
    final_output = tf.reshape(final_output, [future_states.shape[0], future_states.shape[1], 3])

    return final_output

def sample_nll_indices(nlls, k):
    # Subtract by a bit less than the minimum and add a small constant
    nlls = np.array(nlls)
    offset_nlls = nlls - nlls.min() + 0.1

    # Exaggerate differences
    alpha = 3
    adjusted_nlls = np.power(offset_nlls, alpha)

    # Convert adjusted nlls to probabilities
    probabilities = adjusted_nlls / adjusted_nlls.sum()

    # Sample k indices based on the adjusted nll weights
    sampled_indices = np.random.choice(len(nlls), size=k, replace=False, p=probabilities)

    return sampled_indices


def find_similar_traj(value_array, sample, time_steps):

    def calculate_difference(vec1, vec2, weight=0.1):
        """Calculate the Euclidean distance between two vectors."""
        weighted_vec1 = np.array([vec1[0] * weight, vec1[1], vec1[2]])
        weighted_vec2 = np.array([vec2[0] * weight, vec2[1], vec2[2]])

        return np.linalg.norm(weighted_vec1 - weighted_vec2)

    last_two_elements = value_array[-2:]

    differences = []

    for i in range(len(value_array) - time_steps):
        current_difference = 0
        for j in range(2):
            current_difference += calculate_difference(value_array[i + j], last_two_elements[j])

        differences.append(current_difference)

    return np.argpartition(differences, sample)[:sample]+1 # index offset for two elemnts


def create_sdr_basis(res_basis):

    primary_scale = 1 / res_basis
    sub_scale = 1 / (res_basis*res_basis)
    sub_sub_scale = 1 / (res_basis*res_basis*res_basis)

    primary_segment = np.array(range(res_basis)) * primary_scale
    sub_segment = np.array(range(res_basis)) * sub_scale
    sub_sub_segment = np.array(range(res_basis)) * sub_sub_scale

    return np.concatenate([primary_segment, sub_segment, sub_sub_segment])


def create_exploration_actions(sequence_length):
    # Define the actions
    L = [1, 0, 0]
    S = [0, 1, 0]
    R = [0, 0, 1]

    # Function to create a sequence
    def create_sequence(start_action, num_start, end_action):
        return [start_action] * num_start + [end_action] * (sequence_length - num_start)

    # Initialize the action list
    action_list_seq = []

    # Add the all 'S' sequence
    action_list_seq.append([S] * sequence_length)

    # Add the 'R' and 'S' sequences
    for num_R in range(1, sequence_length + 1):
        action_list_seq.append(create_sequence(R, num_R, S))

    # Add the 'L' and 'S' sequences
    for num_L in range(1, sequence_length + 1):
        action_list_seq.append(create_sequence(L, num_L, S))

    # Add the 'S' and 'R' sequences
    for num_R in range(1, sequence_length + 1):
        action_list_seq.append(create_sequence(S, num_R, R))

    # Add the 'S' and 'L' sequences
    for num_L in range(1, sequence_length + 1):
        action_list_seq.append(create_sequence(S, num_L, L))

    # Add the 'L' and 'R' sequences
    for num_R in range(1, sequence_length + 1):
        action_list_seq.append(create_sequence(L, num_R, R))

    # Add the 'R' and 'L' sequences
    for num_L in range(1, sequence_length + 1):
        action_list_seq.append(create_sequence(R, num_L, L))

    return tf.convert_to_tensor(action_list_seq, dtype=tf.float32)

       