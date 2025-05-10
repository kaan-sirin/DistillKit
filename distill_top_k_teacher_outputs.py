import os
import random
import time
from trl import SFTTrainer
from dotenv import load_dotenv
from datasets import load_dataset
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DefaultDataCollator,
)
import wandb
from distill_logits_final import tokenize_function
from distillation_utils import load_config, alpaca_format


class LogitsTrainer(SFTTrainer):
    def __init__(
        self, model, processing_class=None, config=None, debug=False, *args, **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.config = config
        self.processing_class = processing_class
        self.debug = debug  # Print debug information for the first iteration
        self.has_printed_debug_info = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if hasattr(model, "device"):
            device = model.device
        elif hasattr(model, "module"):
            # For DataParallel models, use the device of the module inside
            device = model.module.device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        student_model = model.module if hasattr(model, "module") else model
        
        # Figure out the start index of each response
        start_idxs = []
        for example in inputs["input_ids"]:
            for i, input_id in enumerate(example):
                if input_id == self.processing_class.eos_token_id:  # 128009
                    start_idxs.append(i)
                    break
        
        
        labels = inputs["input_ids"].clone()
        # Shift labels to the left by one position (predict next token)
        labels[:, :-1] = inputs["input_ids"][:, 1:]
        # Set the last position to padding token ID or -100 to be ignored in loss calculation, if answer is 512 tokens long
        labels[:, -1] = -100
        
        def format_token(token_tensor):
                token_val = token_tensor.item()
                if token_val == -100:
                    return '[IGNORE]'
                else:
                    # Attempt to decode, handle potential errors gracefully
                    try:
                        return self.processing_class.decode(token_val)
                    except Exception as e:
                        # Log or handle decoding errors for non -100 tokens if necessary
                        return '[DECODE_ERR]'
        
        # if self.debug:
        #     print(f"\n######################### DEBUG MODE #########################")
        #     print(f"\n# First 20 input tokens of first sequence: {[token.item() for token in inputs['input_ids'][0, :20]]}")
        #     print(f"\n# First 20 input tokens of first sequence, decoded: {[format_token(token) for token in inputs['input_ids'][0, :20]]}")
        #     print(f"\n# First 20 labels of first sequence after cloning and shifting: {[token.item() for token in labels[0, :20]]}")
        #     print(f"\n# First 20 labels of first sequence after cloning and shifting, decoded: {[format_token(token) for token in labels[0, :20]]}")
        #     print(f"\n# Last 20 input tokens of first sequence: {[token.item() for token in inputs['input_ids'][0, -20:]]}")
        #     print(f"\n# Last 20 input tokens of first sequence, decoded: {[format_token(token) for token in inputs['input_ids'][0, -20:]]}")
        #     print(f"\n# Last 20 labels of first sequence after cloning and shifting: {[token.item() for token in labels[0, -20:]]}")
        #     print(f"\n# Last 20 labels of first sequence after cloning and shifting, decoded: {[format_token(token) for token in labels[0, -20:]]}")

        for idx, logits in enumerate(inputs["sparse_logits"]):
            start_idx = start_idxs[idx]
            teacher_len = len(logits)
            
            labels[idx, : start_idx] = -100
            if teacher_len == 512: 
                labels[idx, start_idx + teacher_len +1:] = -100 # Based on my observations
            else:
                labels[idx, start_idx + teacher_len :] = -100
        
        inputs["labels"] = labels
        
        # if self.debug:
        #     start_idx = start_idxs[0]
        #     print(f"\n# First 20 labels of first sequence after masking: {[token.item() for token in labels[0, :20]]}")
        #     print(f"\n# First 20 labels of first sequence after masking, decoded: {[format_token(token) for token in labels[0, :20]]}")
        #     print(f"\n# Last 20 labels of first sequence after masking: {[token.item() for token in labels[0, -20:]]}")
        #     print(f"\n# Last 20 labels of first sequence after masking, decoded: {[format_token(token) for token in labels[0, -20:]]}")
        #     # The input_id token should always be valid, so no check needed there
        #     print(f"\n# 10 tokens before and after the start index ({start_idx} = {self.processing_class.decode(inputs['input_ids'][0, start_idx].item())}) of first sequence: {[token.item() for token in labels[0, start_idx-10:start_idx+10]]}")
        #     print(f"\n# 10 tokens before and after the start index ({start_idx} = {self.processing_class.decode(inputs['input_ids'][0, start_idx].item())}) of first sequence, decoded: {[format_token(token) for token in labels[0, start_idx-10:start_idx+10]]}")
        #     print(f"#############################################################\n")


        

        if self.debug and not self.has_printed_debug_info:
            for i in range(len(inputs["input_ids"])):
                print(f"\n\n######################### DEBUG MODE #########################")
                print(f"# Inputs keys: {inputs.keys()}")
                print(f"# Input IDs shape: {inputs['input_ids'].shape}")
                print(f"# Attention mask shape: {inputs['attention_mask'].shape}")
                number_of_ones = inputs["attention_mask"][i].tolist().count(1)
                print(
                    f"#\n# Number of ones in the {i}-th sequence of the attention mask: {number_of_ones}"
                )
                print(
                    f"# The length of the {i}-th sequence of teacher logits: {len(inputs['sparse_logits'][i])}"
                )

                print(f"# The answer starts at the following index: {start_idxs[i]}, ie. {inputs['input_ids'][i][start_idxs[i]]} = {self.processing_class.decode(inputs['input_ids'][i][start_idxs[i]].item())}")
    
                print(
                    f"#\n# For the {i}-th sequence, the number of ones in the attention mask ({number_of_ones}) should be equal to;"
                )
                print(
                    f"# ->  'prompt length ({start_idxs[i] + 1})' + 'length of the teacher output ({len(inputs['sparse_logits'][i])})' = {start_idxs[i] + 1 + len(inputs['sparse_logits'][i])}"
                )
                if number_of_ones != start_idxs[i] + 1 + len(inputs["sparse_logits"][i]):
                    print(f"#\n# ❌ Values don't match")
                else:
                    print(f"#\n# ✅ Values match")
                print(f"#############################################################\n")

        for idx, logits in enumerate(inputs["sparse_logits"]):
            # Set the labels to -100 for all tokens OTHER than between start_idxs[idx] and start_idxs[idx] + len(logits)
            # My reasoning is that distillation loss is only computed for the output tokens, not input or padding tokens
            # So, cross entropy loss shouldn't be computed for them either
            start_idx = start_idxs[idx]
            teacher_len = len(logits)
            

            if self.debug and not self.has_printed_debug_info:
                print(
                    f"\n######################### DEBUG MODE #########################"
                )
                print(
                    f"Decoded non-masked labels for example {idx} (used for cross entropy loss):"
                )
                for token in inputs["labels"][idx]:
                    if token != -100:
                        print(f"{self.processing_class.decode(int(token))}", end="")
                print(
                    f"\n#############################################################\n"
                )


        sparse_teacher_logits = inputs.pop(
            "sparse_logits"
        )  # Remove the sparse logits from the inputs
        student_outputs = student_model(**inputs)
        
        teacher_lengths = [len(seq) for seq in sparse_teacher_logits]
        

        student_logits = self.match_student_logits_to_teacher(
            student_outputs.logits,
            start_idxs,
            self.config["dataset"]["teacher_data"]["max_new_tokens"],
            teacher_lengths,
        )
        sparse_teacher_logits = self.expand_teacher_sparse_representation(
            sparse_teacher_logits,
            len(self.processing_class),
            self.config["dataset"]["teacher_data"]["max_new_tokens"],
            device=device
        )
        
        if self.debug and not self.has_printed_debug_info:
            print(f"\n######################### DEBUG MODE #########################")
            print(
                f"Decoding and comparing teacher and student logit for the first sequence (after converting them to matching tensors)..."
            )
            
            teacher_active_positions_first_sequence = (
                (sparse_teacher_logits[0] != -1e4).sum(dim=-1) > 0
            ).sum()
            
            if teacher_active_positions_first_sequence > 120:
                print(
                    f"\n -> First teacher sparse logits decoded: {''.join([self.processing_class.decode(sparse_teacher_logits[0][i].argmax()) for i in range(50)])}.....{''.join([self.processing_class.decode(sparse_teacher_logits[0][i].argmax()) for i in range(teacher_active_positions_first_sequence-50, teacher_active_positions_first_sequence)])}"
                )
                print(
                    f"\n -> First student logits decoded: {''.join([self.processing_class.decode(student_logits[0][i].argmax()) for i in range(50)])}.....{''.join([self.processing_class.decode(student_logits[0][i].argmax()) for i in range(teacher_active_positions_first_sequence-50, teacher_active_positions_first_sequence)])}"
                )
            else:
                print(
                    f"\n -> First teacher sparse logits decoded: {''.join([self.processing_class.decode(sparse_teacher_logits[0][i].argmax()) for i in range(teacher_active_positions_first_sequence)])}"
                )
                print(
                    f"\n -> First student logits decoded: {''.join([self.processing_class.decode(student_logits[0][i].argmax()) for i in range(teacher_active_positions_first_sequence)])}"
                )
            print(f"#############################################################\n")
        

        if self.debug and not self.has_printed_debug_info:
            print(f"\n######################### DEBUG MODE #########################")
            print(f"# Analyzing the shapes of student and sparse teacher logits...")
            print(f"# Student logits: {student_logits.shape}")
            print(f"# Sparse teacher logits: {sparse_teacher_logits.shape}")

            print(f"#\n# Value ranges:")
            print(
                f"#   Student logits [min: {student_logits.min():.2f}, max: {student_logits.max():.2f}]"
            )
            print(
                f"#   Teacher logits [min: {sparse_teacher_logits.min():.2f}, max: {sparse_teacher_logits.max():.2f}]"
            )

            # Compare top predictions for first sequence
            print(f"#\n# Top 5 predictions for first token of first sequence:")
            student_top5 = torch.topk(student_logits[0, 0], 5)
            teacher_top5 = torch.topk(sparse_teacher_logits[0, 0], 5)
            print(
                f"#   Student top tokens: {[self.processing_class.decode(idx) for idx in student_top5.indices]}"
            )
            print(
                f"#   Teacher top tokens: {[self.processing_class.decode(idx) for idx in teacher_top5.indices]}"
            )

            # Check how many token positions are non-padding
            student_active_positions = ((student_logits != -1e4).sum(dim=-1) > 0).sum()
            teacher_active_positions = (
                (sparse_teacher_logits != -1e4).sum(dim=-1) > 0
            ).sum()
            print(f"#\n# Number of non-padding token positions:")
            print(f"#   Student: {student_active_positions} positions")
            print(f"#   Teacher: {teacher_active_positions} positions")
            print(f"#############################################################\n")

        total_loss, loss_components = self.distillation_loss(
            student_logits, sparse_teacher_logits, student_outputs.loss
        )

        self._current_loss_kd = loss_components["loss_kd"].detach().item()
        self._current_original_loss = loss_components["original_loss"].detach().item()

        if self.debug and not self.has_printed_debug_info:
            print(f"\n######################### DEBUG MODE #########################")
            print(f"# Knowledge distillation loss: {self._current_loss_kd}")
            print(f"# Cross entropy loss: {self._current_original_loss}")
            print(f"# Total loss: {total_loss}")
            print(f"#############################################################\n")
            self.has_printed_debug_info = True
            exit()

        # initialise at the start of each accumulation cycle
        if self.state.global_step % self.args.gradient_accumulation_steps == 0:
            print(f"Initialising running kd and ce at step {self.state.global_step}")
            self.running_kd = 0.0
            self.running_ce = 0.0
            self.running_tokens = 0
        
        valid_mask = (sparse_teacher_logits != -1e4).any(dim=-1)

        self.running_kd += loss_components["loss_kd"].detach() * valid_mask.sum()
        self.running_ce += loss_components["original_loss"].detach() * valid_mask.sum()
        self.running_tokens += valid_mask.sum()

        return (total_loss, student_outputs) if return_outputs else total_loss

    def expand_teacher_sparse_representation(
        self,
        sparse_logits,
        vocab_size,
        max_seq_length,
        fill_value: float = -1e4,
        device=None
    ):
        """Convert sparse logits representation to a dense tensor.
        The sparse logits are stored as a list of lists of lists of tuples, where the first list is the batch, the second list is the sequence, and the third list is the token logits.
        """
        # Initialize dense tensor with fill_value
        dense_logits = torch.full(
            (len(sparse_logits), max_seq_length, vocab_size), fill_value, device=device
        )

        # Populate dense_logits from sparse_logits
        for i, sequence in enumerate(sparse_logits):
            for pos, token_logits in enumerate(sequence):
                for token_id, logit in token_logits:
                    dense_logits[i, pos, int(token_id)] = logit

        return dense_logits

    def match_student_logits_to_teacher(
        self,
        dense_logits,
        answer_start_indices,
        max_length,
        teacher_lengths,
        fill_value=-1e4,
        debug=False,
    ):
        batch_size, seq_len, vocab_size = dense_logits.shape
        # Pass the device from the input tensor
        device = dense_logits.device 
        padded_logits = torch.full((batch_size, max_length, vocab_size), fill_value, device=device) 

        if debug:
            print(f"\n######################### DEBUG MODE #########################")
            print(f"# Input dense_logits shape: {dense_logits.shape}")
            print(f"# Creating padded_logits with shape: {padded_logits.shape}")
            print(f"# Answer start indices: {answer_start_indices}")

        for i in range(batch_size):
            start_idx = answer_start_indices[i]
            actual_logits = dense_logits[i, start_idx:]
            # Use the teacher length instead of max_length
            actual_length = min(actual_logits.shape[0], teacher_lengths[i])

            if debug:
                print(f"#\n# Batch item {i}:")
                print(f"  # Start index: {start_idx}")
                print(f"  # Teacher length: {teacher_lengths[i]}")
                print(f"  # Extracted logits shape: {actual_logits.shape}")
                print(f"  # Actual length after min(): {actual_length}")

            # Count sequence positions that have any non-padding values
            non_padding_before = (
                ((padded_logits[i] != fill_value).any(dim=-1)).sum().item()
            )
            if debug:
                print(
                    f"# Number of active sequence positions before copy: {non_padding_before}"
                )

            # Only copy the actual number of tokens from teacher
            padded_logits[i, :actual_length] = actual_logits[:actual_length]

            # Count sequence positions that have any non-padding values
            non_padding_after = (
                ((padded_logits[i] != fill_value).any(dim=-1)).sum().item()
            )
            if debug:
                print(
                    f"# Number of active sequence positions after copy: {non_padding_after}"
                )
                print(
                    f"# Value ranges - min: {padded_logits[i].min():.2f}, max: {padded_logits[i].max():.2f}"
                )

            # Check for unexpected active positions
            unexpected = (
                ((padded_logits[i, actual_length:] != fill_value).any(dim=-1))
                .sum()
                .item()
            )
            if unexpected > 0:
                print(
                    f"# WARNING: Found {unexpected} unexpected active positions after position {actual_length}"
                )
        if debug:
            print(f"#############################################################\n")
        return padded_logits

    def distillation_loss(self, student_logits, teacher_logits, original_loss):
        reverse_kld = False
        
        student_logits_scaled = (
            student_logits / self.config["distillation"]["temperature"]
        )
        teacher_logits_scaled = (
            teacher_logits / self.config["distillation"]["temperature"]
        )
        
        # Create mask where valid tokens exist (not padding)
        valid_mask = (teacher_logits != -1e4).any(dim=-1)

        student_distribution = F.softmax(student_logits_scaled, dim=-1)
        teacher_distribution = F.softmax(teacher_logits_scaled, dim=-1)
        
        if reverse_kld:
            log_ps = F.log_softmax(student_logits_scaled, dim=-1)            # log p_s
            ps     = log_ps.exp()                               # p_s
            with torch.no_grad():
                log_pt = F.log_softmax(teacher_logits_scaled, dim=-1)        # log p_t
                log_pt = torch.clamp(log_pt, min=-30)
            token_kl = (ps * (log_ps - log_pt)).sum(-1)         
        
        else:
            # Calculate token-wise KL divergence
            token_kl = F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                teacher_distribution,
                reduction='none'
            ).sum(dim=-1)
        
        # if (self.debug and token_kl.shape[0] == 2) or (self.state.global_step % self.args.gradient_accumulation_steps == 0): # Only if batch size is 2
        #     print_logits = True
        #     print(f"\n\n######################### KL DIVERGENCE #########################")
        #     print(f"# First sequence:")
        #     # Printing in the following format: (student_token, probability), (teacher_token, probability), kl_divergence
        #     kl_div_first = 0
        #     for i in range(len(token_kl[0, valid_mask[0]])):
        #         kl_div_first += token_kl[0, valid_mask[0]][i].item()
        #         if print_logits:
        #             top_3 = torch.topk(teacher_distribution[0, i], 3)
        #             teacher_print = f"[({self.processing_class.decode(top_3.indices[0].item())}, {top_3.values[0].item():.2f}) / ({self.processing_class.decode(top_3.indices[1].item())}, {top_3.values[1].item():.2f}) / ({self.processing_class.decode(top_3.indices[2].item())}, {top_3.values[2].item():.2f})]"
        #             print(f"# Token {i}: ({self.processing_class.decode(student_logits_scaled[0, i].argmax())}, {student_distribution[0, i].max().item():.2f}), {teacher_print}, {token_kl[0, valid_mask[0]][i].item():.2f}")
        #     print(f"# Num of active tokens: {valid_mask[0].sum().item()}")
        #     print(f"# Sum of KL divergence: {kl_div_first}\n\n")
            # TODO: Uncomment this when batch size is 2
            # print(f"#\n# Second sequence:")                 
            # kl_div_second = 0
            # for i in range(len(token_kl[1, valid_mask[1]])):
            #     kl_div_second += token_kl[1, valid_mask[1]][i].item()
            #     if print_logits:
            #         top_3 = torch.topk(teacher_distribution[1, i], 3)
            #         teacher_print = f"[({self.processing_class.decode(top_3.indices[0].item())}, {top_3.values[0].item():.2f}) / ({self.processing_class.decode(top_3.indices[1].item())}, {top_3.values[1].item():.2f}) / ({self.processing_class.decode(top_3.indices[2].item())}, {top_3.values[2].item():.2f})]"
            #         print(f"# Token {i}: ({self.processing_class.decode(student_logits_scaled[1, i].argmax())}, {student_distribution[1, i].max().item():.2f}), {teacher_print}, {token_kl[1, valid_mask[1]][i].item():.2f}")
            # print(f"# Num of active tokens: {valid_mask[1].sum().item()}")
            # print(f"# Sum of KL divergence: {kl_div_second}")
            
            # print(f"#\n# Total mean KL divergence:")
            # print(f"# Mean KL divergence after scaling by temperature: {(kl_div_first + kl_div_second)/(valid_mask[0].sum().item() + valid_mask[1].sum().item()) * (self.config['distillation']['temperature'] ** 2)}")
            # print(f"#############################################################\n")

       
        # Apply mask and normalize by number of actual tokens
        loss_kd = (token_kl * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        loss_kd = loss_kd * (self.config["distillation"]["temperature"] ** 2)

        # total_loss = (
        #     self.config["distillation"]["alpha"] * loss_kd
        #     + (1 - self.config["distillation"]["alpha"]) * original_loss
        # )
        
        # TODO: WARNING, THIS IS AN EXPERIMENT: ALPHA = 1
        total_loss = loss_kd
        return total_loss, {"loss_kd": loss_kd, "original_loss": original_loss}
    
    def log(self, logs, start_time=None):
        # Check if we're in evaluation mode
        is_eval = any(k.startswith("eval_") for k in logs.keys())
        
        # Add component losses for training (keep your existing code)
        if not is_eval and hasattr(self, "_current_loss_kd") and hasattr(self, "_current_original_loss"):
            logs["loss_kd"] = self._current_loss_kd
            logs["original_loss"] = self._current_original_loss
        
        # Add just eval_loss for evaluation
        if is_eval and hasattr(self, "_current_loss_kd") and hasattr(self, "_current_original_loss"):
            # Calculate total loss
            total = (self.config["distillation"]["alpha"] * self._current_loss_kd + 
                   (1 - self.config["distillation"]["alpha"]) * self._current_original_loss)
            logs["eval_loss"] = total

        # Let parent handle logging
        super().log(logs)

        if not is_eval:
            print(f"Running kd: {self.running_kd}, running ce: {self.running_ce}, running tokens: {self.running_tokens}")
            logs["loss_kd"] = (self.running_kd / self.running_tokens).item()
            logs["original_loss"] = (self.running_ce / self.running_tokens).item()


class SparseLogitsCollator(DefaultDataCollator):
    def __call__(self, features):
        # Extract sparse_logits before collation
        sparse_logits = [f.pop("sparse_logits", None) for f in features]

        # Use default collation for everything else
        batch = super().__call__(features)

        # Add sparse_logits back as a list
        batch["sparse_logits"] = sparse_logits

        return batch


if __name__ == "__main__":
    #debug switch
    DEBUG_MODE = False

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    config = load_config()
    # TODO: WARNING, THIS IS AN EXPERIMENT: ALPHA = 1, CHANGE RUN NAME
    run_name = f"alpha_1_distill_teacher_outputs_{config['dataset']['name'].split('/')[-1].replace('-', '_')}_{config['dataset']['num_samples']}_samples_{config['training']['num_train_epochs']}_epochs_{config['dataset']['teacher_data']['top_k']}_top_k_{time.strftime('%m_%d_%H-%M')}"
    # Only create output directory and initialize wandb if not in debug mode
    if not DEBUG_MODE:
        # Output directory
        output_base = config["training"]["output_dir"]
        output_dir = os.path.join(output_base, run_name)
        os.makedirs(output_dir, exist_ok=True)

        # Set up environment
        os.environ["WANDB_PROJECT"] = config["project_name"]

    accelerator = Accelerator()

    # Load and preprocess dataset
    dataset = (
        load_dataset(
            config["dataset"]["name"],
            config["dataset"]["subset"],
            split=config["dataset"]["split"],
        )
        if config["dataset"].get("subset")
        else load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    )

    print(f"WARNING: Filtering out examples where input column isn't empty")
    dataset = dataset.filter(lambda x: x["input"] == "")
    print(f"Filtered dataset to {len(dataset)} examples")


    # Load the sparse logits
    sparse_logits = torch.load(config["dataset"]["teacher_data"]["logits_path"])


    # First take a subset of the dataset if specified, then shuffle
    # This is because I want to know which samples are used for training
    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
        sparse_logits = sparse_logits[:config["dataset"]["num_samples"]]

    print(f"Loaded {len(sparse_logits)} sparse logits")

    if len(sparse_logits) != len(dataset):
        raise ValueError(
            f"Sparse logits and dataset have different lengths: {len(sparse_logits)} != {len(dataset)}"
        )

    # Attach sparse logits to dataset
    original_columns = dataset.column_names
    dataset = dataset.add_column("sparse_logits", sparse_logits)
    print(f"Dataset columns: {dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")

    dataset = dataset.map(
        lambda example: alpaca_format(example, tokenizer),
        remove_columns=original_columns,
        load_from_cache_file=False,
    )

    index = random.randint(0, len(dataset) - 1)
    if DEBUG_MODE:
        print(f"\n######################### DEBUG MODE #########################")
        print(f"Random example - formatted: {dataset[index]['text']}")
        print(f"#############################################################\n")

    # Tokenize the "text" column
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, config["tokenizer"]["max_length"]),
        batched=True,
        num_proc=8,
        load_from_cache_file=False,
    )

    if DEBUG_MODE:
        print(f"\n######################### DEBUG MODE #########################")
        print(
            f"# Random example (keys) - after tokenization: {tokenized_dataset[index].keys()}"
        )
        print(
            f"# Make sure the length of input_ids and attention_mask is enough for input tokens and output tokens.\n# -> Meaning it should be more than the max output tokens: '{config['dataset']['teacher_data']['max_new_tokens']}'"
        )
        print(f"# Length of input_ids: {len(tokenized_dataset[index]['input_ids'])}")
        print(
            f"# Length of attention_mask: {len(tokenized_dataset[index]['attention_mask'])}"
        )
        print(
            f"# Length of sparse logits: {len(tokenized_dataset[index]['sparse_logits'])}"
        )
        print(f"#############################################################\n")

    # Split the dataset into train and test
    test_size = 0.1
    print(f"\nSplitting the dataset into train and test with test size: {test_size}")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size)
    print(f"Train size: {len(tokenized_dataset['train'])}")
    print(f"Test size: {len(tokenized_dataset['test'])}")

    print(f"\nDataset preparation complete. Loading the student model...\n")

    # Load models with configurable flash attention
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], **model_kwargs
    )

    training_arguments = TrainingArguments(
        **config["training"],
        remove_unused_columns=False,
        report_to=["wandb"] if not DEBUG_MODE else [],
    )

    if not DEBUG_MODE:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            config={
                "student_model": config["models"]["student"],
                "distillation_method": config["distillation"]["method"],
                "alpha": config["distillation"]["alpha"],
                "temperature": config["distillation"]["temperature"],
                "num_samples": config["dataset"]["num_samples"],
                "num_epochs": config["training"]["num_train_epochs"],
            },
        )

    trainer = LogitsTrainer(
        model=student_model,
        processing_class=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
        config=config,
        debug=DEBUG_MODE,
        data_collator=SparseLogitsCollator(),
    )

    trainer = accelerator.prepare(trainer)
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    
    # Only save model if not in debug mode
    if not DEBUG_MODE:
        trainer.save_model(output_dir)



# WHAT MIGHT BE THE ISSUE?


#TESTING:
# - Not normalizing the logits (not dividing by max_new_tokens in distillation loss)