import numpy as np
import logging

logger = logging.getLogger(__name__)

# def create_model(args, initial_mean_value, overal_maxlen, vocab):
def create_model(args, initial_mean_value, vocab):
    
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.layers import Bidirectional
    from nea.my_layers import Attention, MeanOverTime, Conv1DWithMasking
    
    ###############################################################################################################################
    ## Recurrence unit type
    #

    if args.recurrent_unit == 'lstm':
        from keras.layers.recurrent import LSTM as RNN
    elif args.recurrent_unit == 'gru':
        from keras.layers.recurrent import GRU as RNN
    elif args.recurrent_unit == 'simple':
        from keras.layers.recurrent import SimpleRNN as RNN

    ###############################################################################################################################
    ## Create Model
    #
    
    # dropout_W = 0.5        # default=0.5
    # dropout_U = 0.1        # default=0.1
    cnn_border_mode='same'
    if initial_mean_value.ndim == 0:
        print("Dim of initial_mean_value is 0")
        initial_mean_value = np.expand_dims(initial_mean_value, axis=1)
    num_outputs = len(initial_mean_value)
    print("Dim of initial_mean_value is:", num_outputs)
    
    if args.model_type == 'cls':
        raise NotImplementedError

    logger.info('Building the model:%s' % args.model_type)
    model = Sequential()

    logger.info('    Adding the Embedding layer')
    model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))
    model.emb_index = 0
    if args.emb_path:
        from nea.w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('        Initializing lookup table')
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        # ipdb.set_trace()
        # model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
        model.layers[model.emb_index].set_weights([emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].get_weights()[0])])
        # ipdb.set_trace()
    logger.info('    Done')

    # Add cnn layer
    if args.cnn_dim > 0:
        logger.info('    Adding the CNN layer')
        logger.info('        cnn_dim:%d' % args.cnn_dim)
        logger.info('        window_size:%d' % args.cnn_window_size)
        model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
        logger.info('    Done')

    # Add LSTM RNN layer
    logger.info('    Adding the LSTM-RNN layer')
    if 'p' in args.model_type:
        layer = RNN(args.rnn_dim, return_sequences=True) #, dropout_W=dropout_W, dropout_U=dropout_U)
    else:
        layer = RNN(args.rnn_dim, return_sequences=False)
    if 'b' in args.model_type:
        # BiLSTM
        logger.info('        Bidirectional layer created!')
        layer = Bidirectional(layer)
    model.add(layer)
    logger.info('    Done')

    # Add MOT or ATT layer
    if 'p' in args.model_type:
        if args.aggregation == 'mot':
            logger.info('    Adding the MOT layer')
            model.add(MeanOverTime(mask_zero=True))
        elif args.aggregation.startswith('att'):
            logger.info('    Adding the ATT layer')
            model.add(Attention(op=args.aggregation, activation='tanh', name='att', init_stdev=0.01))

    model.add(Dense(num_outputs))
    logger.info('    Done')

    model.add(Activation('sigmoid'))
    logger.info('All done!')

    return model
