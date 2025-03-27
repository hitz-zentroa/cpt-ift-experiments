from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import pandas as pd
import json


class PerplexityCalculator:
    def __init__(
        self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the perplexity calculator with a specific model.

        Args:
            model_name (str): HuggingFace model name/path
            device (str): Device to run calculations on ('cuda' or 'cpu')
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def calculate_perplexity(self, source_text, target_text, batch_size=8):
        """
        Calculate perplexity of target text given source text.

        Args:
            source_text (str or list): Source text(s)
            target_text (str or list): Target text(s) to calculate perplexity for
            batch_size (int): Batch size for processing

        Returns:
            float or list: Perplexity score(s)
        """
        # Convert to lists if single strings are provided
        if isinstance(source_text, str):
            source_text = [source_text]
            target_text = [target_text]

        results = []

        # Process in batches
        for i in tqdm(range(0, len(source_text), batch_size),  desc='Batch '):
            batch_source = source_text[i : i + batch_size]
            batch_target = target_text[i : i + batch_size]

            batch_perplexities = self._calculate_batch_perplexity(
                batch_source, batch_target
            )
            results.extend(batch_perplexities)

        return results[0] if len(results) == 1 else results

    def _calculate_batch_perplexity(self, batch_source, batch_target):
        """
        Calculate perplexity for a batch of texts.
        """
        perplexities = []

        with torch.no_grad():
            for src, tgt in zip(batch_source, batch_target):
                # Combine source and target with separator
                combined_text = f"{src} {self.tokenizer.eos_token} {tgt}"

                # Tokenize
                encodings = self.tokenizer(
                    combined_text, return_tensors="pt", truncation=True
                )
                input_ids = encodings.input_ids.to(self.device)

                # Get source length to mask out its contribution
                source_len = len(self.tokenizer.encode(src)) + 1  # +1 for EOS token

                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs.logits

                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # Calculate loss only on target sequence
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                # Reshape loss and mask source sequence
                loss = loss.view(shift_labels.size(0), -1)
                target_loss = loss[:, source_len - 1 :]  # -1 because of shift

                # Calculate perplexity
                mean_nll = target_loss.mean()
                perplexity = torch.exp(mean_nll).item()
                perplexities.append(perplexity)

        return perplexities

    def __call__(self, source_text, target_text, batch_size=8):
        """
        Convenient calling method that wraps calculate_perplexity.
        """
        return self.calculate_perplexity(source_text, target_text, batch_size)


if __name__ == "__main__":

    calculator = PerplexityCalculator(model_name="ahmedselhady/Llama-2-7B-eu-ema")

    data = pd.read_csv(
        "/gscratch5/users/asalem/parallel_structure/sentences_clustered.csv", header=0
    )

    src_sentences = data["text"]
    tgt_sentences = data["target_text"]

    ppls = calculator(tgt_sentences, src_sentences, 100)

    json.dump(
        ppls, open("/gscratch5/users/asalem/parallel_structure/perplexities_flipped.json", "w")
    )
