# Information Maximizing Visual Question Generation

![IQ model](https://cs.stanford.edu/people/ranjaykrishna/iq/model.png)

This repository contains code used to produce the results in the following paper:

### [Information Maximizing Visual Question Generation](https://cs.stanford.edu/people/ranjaykrishna/iq/index.html) <br/>
[Ranjay Krishna](http://ranjaykrishna.com)<sup>, [Michael Bernstein](http://hci.st/msb), [Li Fei-Fei](https://twitter.com/drfeifei) <br/>
IEEE Conference on Computer Vision and Pattern Recognition ([CVPR](http://cvpr2019.thecvf.com/)), 2019 <br/>

If you are using this repository, please use the following citation:

```
@inproceedings{krishna2019information,
  title={Information Maximizing Visual Question Generation},
  author={Krishna, Ranjay and Bernstein, Michael and Fei-Fei, Li },
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Disclaimer:

I have most likely introduced errors while making this public release. Over
time, I will fix the errors.

## Clone the repository and install the dependencies.

You can clone the repository and install the requirements by running the
following:

```
git clone https://github.com/ranjaykrishna/iq.git
cd iq
virtualenv -p python2.7 env
source env/bin/activate
pip install -r requirements.txt
git submodule init
git submodule update
mkdir -p data/processed
```

To download the dataset, [visit our website](https://cs.stanford.edu/people/ranjaykrishna/iq/index.html).

Note that we only distribute the annotations for the answer categories. To
download the images for the VQA dataset, please use the following links:

- [VQA](https://visualqa.org)

## Model training

To train the models, you will need to (1) create a vocabulary object, 
(2) create an `hdf5` dataset with the images, questions and categories,
(3) and then run train and evaluate scripts:

```
# Create the vocabulary file.
python utils/vocab.py

# Create the hdf5 dataset.
python utils/store_dataset.py
python utils/store_dataset.py --output data/processed/iq_val_dataset.hdf5 --questions data/vqa/v2_OpenEnded_mscoco_val2014_questions.json --annotations data/vqa/v2_mscoco_val2014_annotations.json --image-dir data/vqa/val2014

# Train the model.
python train_iq.py

# Evaluate the model.
python evaluate_iq.py
```

This script will train the model and save the weights in the `--model-dir`
directory.  It will also save the configuration parameters in a 
`args.json` file and log events in `train.log`.

However, if you decide that you want more control over the training or
evaluation scripts, check out the instructions below.

## Customized vocabulary creation.

The vocabulary object you create contains <pad>, <start>, <end>, <unk> tokens and decides
which objects to include in the vocabulary and which to consider as <unk>. You can
customize the creation of this vocabulary object using the following options:

```
-h, --help            Show this help message and exit.
--vocab-path          Path for saving vocabulary wrapper.
--questions           Path for train questions file.
--answer-types        Path for the answer types.
--threshold           Minimum word count threshold.
```

## Customized dataset creation.

The dataset creation process can also be customized with the following options:

```
-h, --help            Show this help message and exit.
--image-dir           Directory for resized images.
--vocab-path          Path for saving vocabulary wrapper.
--questions           Path for train annotation file.
--annotations         Path for train annotation file.
--ans2cat             Path for the answer types.
--output              Directory for resized images.
--im_size             Size of images.
--max-q-length        Maximum sequence length for questions.
--max-a-length        Maximum sequence length for answers.
```

## Customized Training.

The model can be trained by calling `python train.py` with the following command
line arguments to modify your training:

```
-h, --help              Show this help message and exit.
  --model-type          [ia2q | via2q | iat2q-type | via2q-type | iq | va2q-
                        type]
  --model-path          Path for saving trained models.
  --crop-size           Size for randomly cropping images.
  --log-step            Step size for prining log info.
  --save-step           Step size for saving trained models.
  --eval-steps          Number of eval steps to run.
  --eval-every-n-steps  Run eval after every N steps.
  --num-epochs 
  --batch-size 
  --num-workers 
  --learning-rate 
  --info-learning-rate 
  --patience 
  --max-examples        For debugging. Limit examples in database.
  --lambda-gen          coefficient to be added in front of the generation
                        loss.
  --lambda-z            coefficient to be added in front of the kl loss.
  --lambda-t            coefficient to be added with the type space loss.
  --lambda-a            coefficient to be added with the answer recon loss.
  --lambda-i            coefficient to be added with the image recon loss.
  --lambda-z-t          coefficient to be added with the t and z space loss.
  --vocab-path          Path for vocabulary wrapper.
  --dataset             Path for train annotation json file.
  --val-dataset         Path for train annotation json file.
  --train-dataset-weights Location of sampling weights for training set.
  --val-dataset-weights Location of sampling weights for training set.
  --cat2name            Location of mapping from category to type name.
  --load-model          Location of where the model weights are.
  --rnn-cell            Type of rnn cell (GRU, RNN or LSTM).
  --hidden-size         Dimension of lstm hidden states.
  --num-layers          Number of layers in lstm.
  --max-length          Maximum sequence length for outputs.
  --encoder-max-len     Maximum sequence length for inputs.
  --bidirectional       Boolean whether the RNN is bidirectional.
  --use-glove           Whether to use GloVe embeddings.
  --embedding-name      Name of the GloVe embedding to use.
  --num-categories      Number of answer types we use.
  --dropout-p           Dropout applied to the RNN model.
  --input-dropout-p     Dropout applied to inputs of the RNN.
  --num-att-layers      Number of attention layers.
  --use-attention       Whether the decoder uses attention.
  --z-size              Dimensions to use for hidden variational space.
  --no-image-recon      Does not try to reconstruct image.
  --no-answer-recon     Does not try to reconstruct answer.
  --no-category-space   Does not try to reconstruct answer.
```

## Customized evaluation

The evaluations can be run using `python evaluate.py` with the following options:

```
-h, --help          Show this help message and exit.
--model-path		Path for loading trained models.
--results-path		Path for saving results.
--preds-path		Path for saving predictions.
--gts-path          Path for saving ground truth.
--batch-size 
--num-workers 
--seed 
--max-examples		When set, only evalutes that many data points.
--num-show          Number of predictions to print.
--from-answer       When set, only evalutes iq model with answers;
					otherwise it tests iq with answer types.
--dataset           Path for train annotation json file.
```

## Contributing.

We welcome everyone to contribute to this reporsitory. Send us a pull request. Feel free to contact me via email or over twitter (@ranjaykrishna).

## License:

The code is under the MIT license. Check `LICENSE` for details.
