# Character-level GPT - 10M Parameters

A character-level language model trained on Moby Dick, based on the GPT architecture.

## Requirements

- Python 3.x
- PyTorch
- `mobydick.txt` data file (should be in the same directory)

## Usage

### Training

Train the model from scratch:

```bash
python char-gpt-10M.py train
```

Resume training from a checkpoint:

```bash
python char-gpt-10M.py train --checkpoint model_checkpoint.pt
```

The training process will:
- Train for 10,000 iterations (or continue from checkpoint)
- Print loss metrics every 1,000 iterations
- Display sample generations during training for reference
- Save model checkpoints to `model_checkpoint.pt`

### Generation

Generate text from a trained model:

```bash
python char-gpt-10M.py generate
```

This will:
- Load the model from `model_checkpoint.pt` (default)
- Generate 2,000 tokens (default)
- Save the output to `generated_text.txt` (default)
- Display a preview of the first 500 characters

### Custom Options

You can customize the generation with command-line arguments:

```bash
python char-gpt-10M.py generate --checkpoint model_checkpoint.pt --tokens 5000 --output my_generated_text.txt
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint file
  - For training: Resume from this checkpoint (optional, starts fresh if not provided)
  - For generation: Load this checkpoint (default: `model_checkpoint.pt`)
- `--tokens`: Number of tokens to generate (default: 2000, for generate mode only)
- `--output`: Output file name for generated text (default: `generated_text.txt`, for generate mode only)

## Model Architecture

- **Parameters**: ~10M
- **Embedding dimension**: 384
- **Number of heads**: 6
- **Number of layers**: 6
- **Block size**: 256
- **Dropout**: 0.2

## Files

- `char-gpt-10M.py`: Main training and generation script
- `mobydick.txt`: Training data
- `model_checkpoint.pt`: Saved model checkpoint (created after training)
- `generated_text.txt`: Generated text output (created after generation)

## Example Workflow

1. **Train the model:**
   ```bash
   python char-gpt-10M.py train
   ```

2. **Generate text:**
   ```bash
   python char-gpt-10M.py generate
   ```

3. **View the generated text:**
   ```bash
   cat generated_text.txt
   ```

## Notes

- The model uses CPU by default. To use MPS (Apple Silicon), uncomment the device line in the script.
- Training may take a while depending on your hardware.
- The model checkpoint includes the vocabulary mappings and iteration number, so you can resume training or generate text even after restarting.
- You can resume training from any checkpoint by using `--checkpoint` with the `train` command.

## Acknowledgments

This project was inspired by [Andrej Karpathy's](https://karpathy.ai/) excellent work on character-level language models, particularly his Tiny Shakespeare GPT implementation and his educational [YouTube video series](https://www.youtube.com/@AndrejKarpathy) on building GPT from scratch. The architecture and training approach are based on his teachings.
