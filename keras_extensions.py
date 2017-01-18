from __future__ import absolute_import
from keras import backend as K


def changing_ndim_rnn(step_function, inputs, initial_states,
                      go_backwards=False, mask=None, constants=None,
                      unroll=False, input_length=None, eliminate_mask_dims=None):
    if K.backend() == 'tensorflow':
        backend_func = changing_ndim_rnn_tf
    else:
        backend_func = changing_ndim_rnn_theano

    return backend_func(step_function, inputs, initial_states,
                        go_backwards, mask, constants, unroll,
                        input_length, eliminate_mask_dims)


def changing_ndim_rnn_theano(step_function, inputs, initial_states, go_backwards, mask,
                             constants, unroll, input_length, eliminate_mask_dims):
    '''Variant of Keras' rnn that allows input's ndim being different from output's
       ndim.

    # Arguments
        inputs: tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape (samples, ...) (no time dimension),
                new_states: list of tensors, same length and shapes
                    as 'states'.
        initial_states: tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape (samples, time),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`scan`).
        input_length: must be specified if using `unroll`.
        eliminate_mask_dims: list of dimension indices that will be eliminated from the mask
                             before applying it to the output.

    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    '''
    from theano import tensor as T
    import theano

    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    if unroll:
        if input_length is None:
            raise Exception('When specifying `unroll=True`, an `input_length` '
                            'must be provided to `rnn`.')

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim-1:
            mask = K.expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, new_states = step_function(inputs[i], states + constants + [mask[i]])

                if len(successive_outputs) == 0:
                    prev_output = K.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                if eliminate_mask_dims is not None:
                    output_mask = K.sum(mask[i], axis=eliminate_mask_dims)
                else:
                    output_mask = mask[i]

                output = T.switch(output_mask, output, prev_output)
                kept_states = []
                for state, new_state in zip(states, new_states):
                    kept_states.append(T.switch(output_mask, new_state, state))
                states = kept_states

                successive_outputs.append(output)
                successive_states.append(states)

            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_output = T.zeros_like(step_function(inputs[0], initial_states + constants + [mask[0]])[0])
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)

            def _step(input, mask, output_tm1, *states):
                output, new_states = step_function(input, states + (mask,))
                if eliminate_mask_dims is not None:
                    output_mask = K.sum(mask, axis=eliminate_mask_dims)
                else:
                    output_mask = mask
                # output previous output if masked.
                output = T.switch(output_mask, output, output_tm1)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(output_mask, new_state, state))
                return [output] + return_states

            results, _ = theano.scan(
                _step,
                sequences=[inputs, mask],
                outputs_info=[initial_output] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if type(results) is list:
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []
    else:
        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, states = step_function(inputs[i], states + (None,))  # None for mask.
                successive_outputs.append(output)
                successive_states.append(states)
            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))

        else:
            def _step(input, *states):
                output, new_states = step_function(input, states + (None,))  # None for mask.
                return [output] + new_states

            results, _ = theano.scan(
                _step,
                sequences=inputs,
                outputs_info=[None] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if type(results) is list:
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, states


def changing_ndim_rnn_tf(step_function, inputs, initial_states, go_backwards, mask,
                         constants, unroll, input_length, eliminate_mask_dims):
    '''Iterates over the time dimension of a tensor.

    # Arguments
        inputs: tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape (samples, output_dim) (no time dimension),
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        initial_states: tensor with shape (samples, output_dim) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape (samples, time, 1),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: with TensorFlow the RNN is always unrolled, but with Theano you
            can use this boolean flag to unroll the RNN.
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple (last_output, outputs, new_states).

        last_output: the latest output of the rnn, of shape (samples, ...)
        outputs: tensor with shape (samples, time, ...) where each
            entry outputs[s, t] is the output of the step function
            at time t for sample s.
        new_states: list of tensors, latest states returned by
            the step function, of shape (samples, ...).
    '''
    import tensorflow as tf

    ndim = len(inputs.get_shape())
    assert ndim >= 3, 'Input should be at least 3D.'
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))

    if constants is None:
        constants = []

    if unroll:
        if not inputs.get_shape()[0]:
            raise Exception('Unrolling requires a fixed number of timesteps.')

        states = initial_states
        successive_states = []
        successive_outputs = []

        input_list = tf.unpack(inputs)
        if go_backwards:
            input_list.reverse()

        if mask is not None:
            # Transpose not supported by bool tensor types, hence round-trip to uint8.
            mask = tf.cast(mask, tf.uint8)
            if len(mask.get_shape()) == ndim - 1:
                mask = K.expand_dims(mask)
            # Reshaping mask to make timesteps the first dimension.
            mask = tf.cast(tf.transpose(mask, axes), tf.bool)
            mask_list = tf.unpack(mask)

            if go_backwards:
                mask_list.reverse()

            # Iterating over timesteps.
            for input, mask_t in zip(input_list, mask_list):
                # Changing ndim modification: Pass the mask to the step function as a constant.
                output, new_states = step_function(input, states + constants + [mask_t])

                # tf.select needs its condition tensor to be the same shape as its two
                # result tensors, but in our case the condition (mask) tensor is
                # (nsamples, 1), and A and B are (nsamples, ndimensions). So we need to
                # broadcast the mask to match the shape of A and B. That's what the
                # tile call does, is just repeat the mask along its second dimension
                # ndimensions times.
                output_mask_t = tf.tile(mask_t, tf.pack(([1] * (ndim-2)) + [tf.shape(output)[1]]))

                if len(successive_outputs) == 0:
                    prev_output = K.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                # Changing ndim modification: Define output mask with appropriate dims eliminated.
                if eliminate_mask_dims is not None:
                    output_mask_t = tf.cast(K.any(output_mask_t, axis=eliminate_mask_dims), tf.bool)
                else:
                    output_mask_t = tf.cast(output_mask_t, tf.bool)

                output = tf.select(output_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    state_mask_t = tf.tile(mask_t, tf.pack(([1] * (ndim-2)) + [tf.shape(new_state)[1]]))
                    # Changing ndim modification: Define output mask with appropriate dims eliminated.
                    if eliminate_mask_dims is not None:
                        state_mask_t = tf.cast(K.any(state_mask_t, axis=eliminate_mask_dims), tf.bool)
                    else:
                        state_mask_t = tf.cast(state_mask_t, tf.bool)
                    return_states.append(tf.select(state_mask_t, new_state, state))

                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = tf.pack(successive_outputs)
        else:
            for input in input_list:
                output, states = step_function(input, states + constants + [None])  # None for mask
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.pack(successive_outputs)

    else:
        from tensorflow.python.ops.rnn import _dynamic_rnn_loop

        if go_backwards:
            inputs = tf.reverse(inputs, [True] + [False] * (ndim - 1))

        states = initial_states
        nb_states = len(states)
        if nb_states == 0:
            # use dummy state, otherwise _dynamic_rnn_loop breaks
            state = inputs[:, 0, :]
            state_size = state.get_shape()[-1]
        else:
            state_size = int(states[0].get_shape()[-1])
            if nb_states == 1:
                state = states[0]
            else:
                state = tf.concat(1, states)

        if mask is not None:
            if len(initial_states) == 0:
                raise ValueError('No initial states provided! '
                                 'When using masking in an RNN, you should '
                                 'provide initial states '
                                 '(and your step function should return '
                                 'as its first state at time `t` '
                                 'the output at time `t-1`).')
            if go_backwards:
                mask = tf.reverse(mask, [True] + [False] * (ndim - 2))

            # Transpose not supported by bool tensor types, hence round-trip to uint8.
            mask = tf.cast(mask, tf.uint8)
            if len(mask.get_shape()) == ndim - 1:
                mask = K.expand_dims(mask)
            mask = tf.transpose(mask, axes)
            # Concatenate at the last dim.
            inputs = tf.concat(ndim-1, [tf.cast(mask, inputs.dtype), inputs])

            def _step(input, state):
                if nb_states > 1:
                    states = []
                    for i in range(nb_states):
                        states.append(state[:, i * state_size: (i + 1) * state_size])
                else:
                    states = [state]

                # The time dimension is not present here.
                step_ndim = ndim - 1
                # Permuting only to take out the mask.
                permuted_input = K.permute_dimensions(input, (step_ndim-1,) + tuple(range(step_ndim-1)))
                mask_t = K.expand_dims(permuted_input[0])
                permuted_input = permuted_input[1:]
                input = K.permute_dimensions(permuted_input, tuple(range(1, step_ndim)) + (0,))
                # changing ndim fix: eliminate necessary dims after selecting the mask from the input.
                if eliminate_mask_dims is not None:
                    output_mask_t = K.sum(mask_t, axis=eliminate_mask_dims)

                mask_t = tf.cast(mask_t, tf.bool)
                output_mask_t = tf.cast(output_mask_t, tf.bool)
                
                output, new_states = step_function(input, states + constants + [mask_t])

                tiled_output_mask_t = tf.tile(output_mask_t, tf.pack([1, tf.shape(output)[1]]))
                output = tf.select(tiled_output_mask_t, output, states[0])

                return_states = []
                for state, new_state in zip(states, new_states):
                    tiled_state_mask_t = tf.tile(output_mask_t, tf.pack([1, tf.shape(state)[1]]))
                    return_states.append(tf.select(tiled_state_mask_t, new_state, state))

                if len(return_states) == 1:
                    new_state = return_states[0]
                else:
                    new_state = tf.concat(1, return_states)

                return output, new_state
        else:
            def _step(input, state):
                if nb_states > 1:
                    states = []
                    for i in range(nb_states):
                        states.append(state[:, i * state_size: (i + 1) * state_size])
                elif nb_states == 1:
                    states = [state]
                else:
                    states = []
                output, new_states = step_function(input, states + constants + [None])  # None for mask

                if len(new_states) > 1:
                    new_state = tf.concat(1, new_states)
                elif len(new_states) == 1:
                    new_state = new_states[0]
                else:
                    # return dummy state, otherwise _dynamic_rnn_loop breaks
                    new_state = output
                return output, new_state

        _step.state_size = state_size * nb_states
        # recover output size by calling _step on the first input
        slice_begin = tf.pack([0] * ndim)
        slice_size = tf.pack([1] + [-1] * (ndim - 1))
        first_input = tf.slice(inputs, slice_begin, slice_size)
        first_input = tf.squeeze(first_input, [0])
        _step.output_size = int(_step(first_input, state)[0].get_shape()[-1])

        (outputs, final_state) = _dynamic_rnn_loop(
            _step,
            inputs,
            state,
            parallel_iterations=32,
            swap_memory=True,
            sequence_length=None)

        if nb_states > 1:
            new_states = []
            for i in range(nb_states):
                new_states.append(final_state[:, i * state_size: (i + 1) * state_size])
        elif nb_states == 1:
            new_states = [final_state]
        else:
            new_states = []

        outputs_ndim = len(outputs.get_shape())
        # all this circus is to recover the last vector in the sequence.
        slice_begin = tf.pack([tf.shape(outputs)[0] - 1] + [0] * (outputs_ndim - 1))
        slice_size = tf.pack([1] + [-1] * (outputs_ndim - 1))
        last_output = tf.slice(outputs, slice_begin, slice_size)
        last_output = tf.squeeze(last_output, [0])

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states


def switch(condition, then_tensor, else_tensor):
    """
    Keras' implementation of switch for tensorflow uses tf.switch which accepts only scalar conditions.
    It should use tf.select instead.
    """
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        condition_shape = condition.get_shape()
        input_shape = then_tensor.get_shape()
        if condition_shape[-1] != input_shape[-1] and condition_shape[-1] == 1:
            # This means the last dim is an embedding dim. Keras does not mask this dimension. But tf wants
            # the condition and the then and else tensors to be the same shape.
            condition = K.dot(tf.cast(condition, tf.float32), tf.ones((1, input_shape[-1])))
        return tf.select(tf.cast(condition, dtype=tf.bool), then_tensor, else_tensor)
    else:
        import theano.tensor as T
        return T.switch(condition, then_tensor, else_tensor)
