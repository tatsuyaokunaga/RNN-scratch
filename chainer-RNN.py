from chainer.datasets import tuple_dataset
import text_datasets
import chainer
from chainer import training
from chainer.training import extensions
import text_datasets
import nets
from nlp_utils import convert_seq

def main():
    
    args={
        'gpu':-1,
        'dataset': 'imdb.binary',
        'model': 'rnn',
        'batchsize': 64,
        'epoch': 3,
        'out': 'result',
        'unit': 100,
        'layer':1,
        'dropout':0.4,
        'char_based': False
    }

    # Load a dataset
    if args['dataset'] == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            char_based=args['char_based'])
    elif args['dataset'].startswith('imdb.'):
        print("IMDB datasets")
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=args['dataset'].endswith('.fine'),
            char_based=args['char_based'])
    elif args['dataset'] in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            args['dataset'], char_based=args['char_based'])

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))
    

    train_iter = chainer.iterators.SerialIterator(train[:1000], args['batchsize'])
    test_iter = chainer.iterators.SerialIterator(test[:1000], args['batchsize'],
                                                 repeat=False, shuffle=False)

    # return train_iter, test_iter
    # Setup a model
    if args['model'] == 'rnn':
        Encoder = nets.RNNEncoder
        print(type(Encoder))
    elif args['model'] == 'cnn':
        Encoder = nets.CNNEncoder
    elif args['model'] == 'bow':
        Encoder = nets.BOWMLPEncoder

    encoder = Encoder(n_layers=args['layer'], n_vocab=len(vocab),
                      n_units=args['unit'], dropout=args['dropout'])
    model = nets.TextClassifier(encoder, n_class)
    if args['gpu'] >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args['gpu']).use()
        model.to_gpu()  # Copy the model to the GPU
    

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))


    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args['gpu'])
    trainer = training.Trainer(updater, (args['epoch'], 'epoch'), out=args['out'])

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        test_iter, model,
        converter=convert_seq, device=args['gpu']))

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    print("STRAT Training!")
    # Run the training
    trainer.run()
    print("Finished!")


if __name__ == '__main__':
    main()
