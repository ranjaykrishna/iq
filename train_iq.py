"""This script is used to VAE question generation.
"""

from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import argparse
import json
import logging
import os
import random
import time
import torch
import torch.nn as nn

from models import IQ
from utils import Vocabulary
from utils import get_glove_embedding
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import gaussian_KL_loss


def create_model(args, vocab, embedding=None):
    """Creates the model.

    Args:
        args: Instance of Argument Parser.
        vocab: Instance of Vocabulary.

    Returns:
        An IQ model.
    """
    # Load GloVe embedding.
    if args.use_glove:
        embedding = get_glove_embedding(args.embedding_name,
                                        args.hidden_size,
                                        vocab)
    else:
        embedding = None

    # Build the models
    logging.info('Creating IQ model...')
    vqg = IQ(len(vocab), args.max_length, args.hidden_size,
             args.num_categories,
             vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
             num_layers=args.num_layers,
             rnn_cell=args.rnn_cell,
             dropout_p=args.dropout_p,
             input_dropout_p=args.input_dropout_p,
             encoder_max_len=args.encoder_max_len,
             embedding=embedding,
             num_att_layers=args.num_att_layers,
             z_size=args.z_size,
             no_answer_recon=args.no_answer_recon,
             no_image_recon=args.no_image_recon,
             no_category_space=args.no_category_space)
    return vqg


def evaluate(vqg, data_loader, criterion, l2_criterion, args):
    """Calculates vqg average loss on data_loader.

    Args:
        vqg: question generation model.
        data_loader: Iterator for the data.
        criterion: The loss function used to evaluate the loss.
        l2_criterion: The loss function used to evaluate the l2 loss.
        args: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    vqg.eval()
    total_gen_loss = 0.0
    total_kl = 0.0
    total_recon_image_loss = 0.0
    total_recon_answer_loss = 0.0
    total_z_t_kl = 0.0
    total_t_kl_loss = 0.0
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    for iterations, (images, questions, answers,
            categories, qindices) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            categories = categories.cuda()
            qindices = qindices.cuda()
        alengths = process_lengths(answers)

        # Forward, Backward and Optimize
        image_features = vqg.encode_images(images)
        answer_features = vqg.encode_answers(answers, alengths)
        mus, logvars = vqg.encode_into_z(image_features, answer_features)
        zs = vqg.reparameterize(mus, logvars)
        (outputs, _, other) = vqg.decode_questions(
                image_features, zs, questions=questions,
                teacher_forcing_ratio=1.0)

        # Reorder the questions based on length.
        questions = torch.index_select(questions, 0, qindices)

        # Ignoring the start token.
        questions = questions[:, 1:]
        qlengths = process_lengths(questions)

        # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
        # (BATCH x MAX_LEN x VOCAB).
        outputs = [o.unsqueeze(1) for o in outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = torch.index_select(outputs, 0, qindices)

        # Calculate the loss.
        targets = pack_padded_sequence(questions, qlengths,
                                       batch_first=True)[0]
        outputs = pack_padded_sequence(outputs, qlengths,
                                       batch_first=True)[0]
        gen_loss = criterion(outputs, targets)
        total_gen_loss += gen_loss.data.item()

        # Get KL loss if it exists.
        kl_loss = gaussian_KL_loss(mus, logvars)
        total_kl += kl_loss.item()

        # Category.
        if not args.no_category_space:
            category_features = vqg.encode_categories(categories)
            t_mus, t_logvars = vqg.encode_into_t(
                    image_features, category_features)
            t_kl = gaussian_KL_loss(t_mus, t_logvars)
            total_t_kl_loss += t_kl.item()
            z_t_kl = compute_two_gaussian_loss(
                    mus, logvars, t_mus, t_logvars)
            total_z_t_kl += z_t_kl.item()

        # Reconstruction.
        if not args.no_image_recon or not args.no_answer_recon:
            image_targets = image_features.detach()
            answer_targets = answer_features.detach()
            recon_image_features, recon_answer_features = vqg.reconstruct_inputs(
                    image_targets, answer_targets)
            if not args.no_image_recon:
                recon_i_loss = l2_criterion(recon_image_features, image_targets)
                total_recon_image_loss += recon_i_loss.item()
            if not args.no_answer_recon:
                recon_a_loss = l2_criterion(recon_answer_features, answer_targets)
                total_recon_answer_loss += recon_a_loss.item()

        # Quit after eval_steps.
        if args.eval_steps is not None and iterations >= args.eval_steps:
            break

        # Print logs
        if iterations % args.log_step == 0:
             delta_time = time.time() - start_time
             start_time = time.time()
             logging.info('Time: %.4f, Step [%d/%d], gen loss: %.4f, '
                          'KL: %.4f, I-recon: %.4f, A-recon: %.4f, '
                          'z-t-KL: %.4f, t-KL: %.4f'
                         % (delta_time, iterations, total_steps,
                            total_gen_loss/(iterations+1),
                            total_kl/(iterations+1),
                            total_recon_image_loss/(iterations+1),
                            total_recon_answer_loss/(iterations+1),
                            total_z_t_kl/(iterations+1),
                            total_t_kl_loss/(iterations+1)))
    total_info_loss = total_recon_image_loss + total_recon_answer_loss
    return total_gen_loss / (iterations+1), total_info_loss / (iterations + 1)


def run_eval(vqg, data_loader, criterion, l2_criterion, args, epoch,
             scheduler, info_scheduler):
    logging.info('=' * 80)
    start_time = time.time()
    val_gen_loss, val_info_loss = evaluate(
            vqg, data_loader, criterion, l2_criterion, args)
    delta_time = time.time() - start_time
    scheduler.step(val_gen_loss)
    scheduler.step(val_info_loss)
    logging.info('Time: %.4f, Epoch [%d/%d], Val-gen-loss: %.4f, '
                 'Val-info-loss: %.4f' % (
        delta_time, epoch, args.num_epochs, val_gen_loss, val_info_loss))
    logging.info('=' * 80)


def sample_for_each_category(vqg, image, args):
    """Sample a question per category.

    Args:
        vqg: Question generation model.
        image: The image for which to generate questions for.
        args: Instance of ArgumentParser.

    Returns:
        A list of questions per category.
    """
    if args.no_category_space:
        return None
    categories = torch.LongTensor(range(args.num_categories))
    if torch.cuda.is_available():
        categories = categories.cuda()
    images = image.unsqueeze(0).expand((
        args.num_categories, image.size(0), image.size(1), image.size(2)))
    outputs = vqg.predict_from_category(images, categories)
    return outputs


def compare_outputs(images, questions, answers, categories,
                    alengths, vqg, vocab, logging, cat2name,
                    args, num_show=5):
    """Sanity check generated output as we train.

    Args:
        images: Tensor containing images.
        questions: Tensor containing questions as indices.
        answers: Tensor containing answers as indices.
        categories: Tensor containing categories as indices.
        alengths: list of answer lengths.
        vqg: A question generation instance.
        vocab: An instance of Vocabulary.
        logging: logging to use to report results.
        cat2name: Mapping from category index to answer type name.
    """
    vqg.eval()

    # Forward pass through the model.
    outputs = vqg.predict_from_answer(images, answers, lengths=alengths)

    for _ in range(num_show):
        logging.info("         ")
        i = random.randint(0, images.size(0) - 1)  # Inclusive.

        # Sample some types.
        if not args.no_category_space:
            category_outputs = vqg.predict_from_category(images, categories)
            category_question = vocab.tokens_to_words(category_outputs[i])
            logging.info('Typed question: %s' % category_question)
            category_checks = sample_for_each_category(vqg, images[i], args)
            category_checks = [cat2name[idx] + ': ' + vocab.tokens_to_words(category_checks[j])
                           for idx, j in enumerate(range(category_checks.size(0)))]
            logging.info('category checks: ' + ', '.join(category_checks))

        # Log the outputs.
        output = vocab.tokens_to_words(outputs[i])
        question = vocab.tokens_to_words(questions[i])
        answer = vocab.tokens_to_words(answers[i])
        logging.info('Sampled question : %s\n'
                     'Target  question (%s): %s -> %s'
                     % (output, cat2name[categories[i].item()],
                        question, answer))
        logging.info("         ")


def compute_two_gaussian_loss(mu1, logvar1, mu2, logvar2):
    """Computes the KL loss between the embedding attained from the answers
    and the categories.

    KL divergence between two gaussians:
        log(sigma_2/sigma_1) + (sigma_2^2 + (mu_1 - mu_2)^2)/(2sigma_1^2) - 0.5

    Args:
        mu1: Means from first space.
        logvar1: Log variances from first space.
        mu2: Means from second space.
        logvar2: Means from second space.
    """
    numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
    fraction = torch.div(numerator, (logvar2.exp() + 1e-8))
    kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1)
    return kl / (mu1.size(0) + 1e-8)


def train(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Save the arguments.
    with open(os.path.join(args.model_path, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_path, 'train.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)

    # Load the category types.
    cat2name = json.load(open(args.cat2name))

    # Build data loader
    logging.info("Building data loader...")
    train_sampler = None
    val_sampler = None
    if os.path.exists(args.train_dataset_weights):
        train_weights = json.load(open(args.train_dataset_weights))
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_weights, len(train_weights))
    if os.path.exists(args.val_dataset_weights):
        val_weights = json.load(open(args.val_dataset_weights))
        val_weights = torch.DoubleTensor(val_weights)
        val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                val_weights, len(val_weights))
    data_loader = get_loader(args.dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 sampler=train_sampler)
    val_data_loader = get_loader(args.val_dataset, transform,
                                     args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     max_examples=args.max_examples,
                                     sampler=val_sampler)
    logging.info("Done")

    vqg = create_model(args, vocab)
    if args.load_model is not None:
        vqg.load_state_dict(torch.load(args.load_model))
    logging.info("Done")

    # Loss criterion.
    pad = vocab(vocab.SYM_PAD)  # Set loss weight for 'pad' symbol to 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    l2_criterion = nn.MSELoss()

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        vqg.cuda()
        criterion.cuda()
        l2_criterion.cuda()

    # Parameters to train.
    gen_params = vqg.generator_parameters()
    info_params = vqg.info_parameters()
    learning_rate = args.learning_rate
    info_learning_rate = args.info_learning_rate
    gen_optimizer = torch.optim.Adam(gen_params, lr=learning_rate)
    info_optimizer = torch.optim.Adam(info_params, lr=info_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=True, min_lr=1e-7)
    info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
                                       factor=0.1, patience=args.patience,
                                       verbose=True, min_lr=1e-7)

    # Train the model.
    total_steps = len(data_loader)
    start_time = time.time()
    n_steps = 0

    # Optional losses. Initialized here for logging.
    recon_answer_loss = 0.0
    recon_image_loss = 0.0
    kl_loss = 0.0
    z_t_kl = 0.0
    t_kl = 0.0
    for epoch in range(args.num_epochs):
        for i, (images, questions, answers,
                categories, qindices) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
                categories = categories.cuda()
                qindices = qindices.cuda()
            alengths = process_lengths(answers)

            # Eval now.
            if (args.eval_every_n_steps is not None and
                    n_steps >= args.eval_every_n_steps and
                    n_steps % args.eval_every_n_steps == 0):
                run_eval(vqg, val_data_loader, criterion, l2_criterion,
                         args, epoch, scheduler, info_scheduler)
                compare_outputs(images, questions, answers, categories,
                                alengths, vqg, vocab, logging, cat2name, args)

            # Forward.
            vqg.train()
            gen_optimizer.zero_grad()
            info_optimizer.zero_grad()
            image_features = vqg.encode_images(images)
            answer_features = vqg.encode_answers(answers, alengths)

            # Question generation.
            mus, logvars = vqg.encode_into_z(image_features, answer_features)
            zs = vqg.reparameterize(mus, logvars)
            (outputs, _, _) = vqg.decode_questions(
                    image_features, zs, questions=questions,
                    teacher_forcing_ratio=1.0)

            # Reorder the questions based on length.
            questions = torch.index_select(questions, 0, qindices)

            # Ignoring the start token.
            questions = questions[:, 1:]
            qlengths = process_lengths(questions)

            # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
            # (BATCH x MAX_LEN x VOCAB).
            outputs = [o.unsqueeze(1) for o in outputs]
            outputs = torch.cat(outputs, dim=1)
            outputs = torch.index_select(outputs, 0, qindices)

            # Calculate the generation loss.
            targets = pack_padded_sequence(questions, qlengths,
                                           batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, qlengths,
                                           batch_first=True)[0]
            gen_loss = criterion(outputs, targets)
            total_loss = 0.0
            total_loss += args.lambda_gen * gen_loss
            gen_loss = gen_loss.item()

            # Variational loss.
            kl_loss = gaussian_KL_loss(mus, logvars)
            total_loss += args.lambda_z * kl_loss
            kl_loss = kl_loss.item()

            # Category regularization.
            if not args.no_category_space:
                category_features = vqg.encode_categories(categories)
                t_mus, t_logvars = vqg.encode_into_t(
                        image_features, category_features)
                t_kl = gaussian_KL_loss(t_mus, t_logvars)
                total_loss += args.lambda_t * t_kl
                t_kl = t_kl.item()
                z_t_kl = compute_two_gaussian_loss(
                        mus, logvars, t_mus, t_logvars)
                total_loss += args.lambda_z_t * z_t_kl
                z_t_kl = z_t_kl.item()

            # Generator Backprop.
            total_loss.backward()
            gen_optimizer.step()

            # Reconstruction loss.
            recon_image_loss = 0.0
            recon_answer_loss = 0.0
            if not args.no_answer_recon or not args.no_image_recon:
                total_info_loss = 0.0
                gen_optimizer.zero_grad()
                info_optimizer.zero_grad()
                answer_targets = answer_features.detach()
                image_targets = image_features.detach()
                recon_image_features, recon_answer_features = vqg.reconstruct_inputs(
                        image_targets, answer_targets)

                # Answer reconstruction loss.
                if not args.no_answer_recon:
                    recon_a_loss = l2_criterion(recon_answer_features, answer_targets)
                    total_info_loss += args.lambda_a * recon_a_loss
                    recon_answer_loss = recon_a_loss.item()

                # Image reconstruction loss.
                if not args.no_image_recon:
                    recon_i_loss = l2_criterion(recon_image_features, image_targets)
                    total_info_loss += args.lambda_i * recon_i_loss
                    recon_image_loss = recon_i_loss.item()

                # Info backprop.
                total_info_loss.backward()
                info_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                             'LR: %f, gen: %.4f, KL: %.4f, '
                             'I-recon: %.4f, A-recon: %.4f, z-t-KL: %.4f, '
                             't-KL: %.4f'
                             % (delta_time, epoch, args.num_epochs, i,
                                total_steps, gen_optimizer.param_groups[0]['lr'],
                                gen_loss, kl_loss,
                                recon_image_loss, recon_answer_loss,
                                z_t_kl, t_kl))

            # Save the models
            if args.save_step is not None and (i+1) % args.save_step == 0:
                torch.save(vqg.state_dict(),
                           os.path.join(args.model_path,
                                        'vqg-tf-%d-%d.pkl'
                                        % (epoch + 1, i + 1)))

        torch.save(vqg.state_dict(),
                   os.path.join(args.model_path,
                                'vqg-tf-%d.pkl' % (epoch+1)))

        # Evaluation and learning rate updates.
        run_eval(vqg, val_data_loader, criterion, l2_criterion,
                 args, epoch, scheduler, info_scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='weights/tf1/',
                        help='Path for saving trained models')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for prining log info')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=300,
                        help='Run eval after every N steps.')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--info-learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Lambda values.
    parser.add_argument('--lambda-gen', type=float, default=1.0,
                        help='coefficient to be added in front of the generation loss.')
    parser.add_argument('--lambda-z', type=float, default=0.001,
                        help='coefficient to be added in front of the kl loss.')
    parser.add_argument('--lambda-t', type=float, default=0.0001,
                        help='coefficient to be added with the type space loss.')
    parser.add_argument('--lambda-a', type=float, default=0.001,
                        help='coefficient to be added with the answer recon loss.')
    parser.add_argument('--lambda-i', type=float, default=0.001,
                        help='coefficient to be added with the image recon loss.')
    parser.add_argument('--lambda-z-t', type=float, default=0.001,
                        help='coefficient to be added with the t and z space loss.')

    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='data/processed/iq_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/processed/iq_val_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='data/processed/iq_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='data/processed/iq_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')


    # Model parameters
    parser.add_argument('--rnn-cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers in lstm.')
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum sequence length for outputs.')
    parser.add_argument('--encoder-max-len', type=int, default=4,
                        help='Maximum sequence length for inputs.')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use-glove', action='store_true',
                        help='Whether to use GloVe embeddings.')
    parser.add_argument('--embedding-name', type=str, default='6B',
                        help='Name of the GloVe embedding to use.')
    parser.add_argument('--num-categories', type=int, default=16,
                        help='Number of answer types we use.')
    parser.add_argument('--dropout-p', type=float, default=0.3,
                        help='Dropout applied to the RNN model.')
    parser.add_argument('--input-dropout-p', type=float, default=0.3,
                        help='Dropout applied to inputs of the RNN.')
    parser.add_argument('--num-att-layers', type=int, default=2,
                        help='Number of attention layers.')
    parser.add_argument('--z-size', type=int, default=100,
                        help='Dimensions to use for hidden variational space.')

    # Ablations.
    parser.add_argument('--no-image-recon', action='store_true', default=False,
                        help='Does not try to reconstruct image.')
    parser.add_argument('--no-answer-recon', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no-category-space', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')

    args = parser.parse_args()
    train(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
