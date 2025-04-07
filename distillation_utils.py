from dotenv import load_dotenv
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import numpy as np
import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def medqa_format(example):
    try:
        text = (
            "Du är en medicinsk expert. Svara på följande flervalsfråga. \n\n"
            f"{example['question']}\n\n"
            f"{example['options']}\n\n"
            f"{example['model_response']}"
        )

        return {"text": text}

    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def  medlfqa_format(example):
    try: 
        text = (
            f"Question: {example['Question']}\n\n"
            f"Answer: {example['Free_form_answer']}"
        )
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if config.get("training", {}).get("learning_rate"):
                config["training"]["learning_rate"] = float(
                    config["training"]["learning_rate"]
                )
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Program will exit.")
        raise  # Still crash, but with a clearer message
    except yaml.YAMLError as e:
        print(f"Error parsing {config_path}: {e}")
        raise


class LivePlotCallback(TrainerCallback):
    def __init__(
        self,
        plot_path="./loss_plot.png",
        update_freq=20,
        moving_avg_window=10,
        distillation_method=None,
        config=None,
    ):
        self.losses = []
        self.loss_kds = []
        self.original_losses = []
        self.learning_rates = []
        self.steps = []
        self.plot_path = plot_path
        self.update_freq = update_freq
        self.moving_avg_window = moving_avg_window  # Make window size configurable
        self.distillation_method = distillation_method
        self.config = config
        # Validation metrics
        self.eval_losses = []
        self.eval_steps = []

        # For accumulating losses over gradient accumulation steps
        self.accumulated_loss_kd = 0
        self.accumulated_original_loss = 0
        self.accumulation_count = 0
        self.current_step = -1

        self.epoch_end_steps = []
        self.last_epoch = -1

    def on_train_end(self, args, state, control, **kwargs):
        if self.accumulation_count > 0:
            # Calculate proper averages
            avg_loss_kd = self.accumulated_loss_kd / self.accumulation_count
            avg_original_loss = self.accumulated_original_loss / self.accumulation_count

            # Only append if it's actually new data
            if self.current_step > self.steps[-1]:
                self.losses.append(
                    state.log_history[-1].get("loss") if state.log_history else None
                )
                self.steps.append(self.current_step)
                self.loss_kds.append(avg_loss_kd)
                self.original_losses.append(avg_original_loss)
                self.learning_rates.append(
                    self.learning_rates[-1] if self.learning_rates else 0
                )

        self.update_plot()
        print(f"Final plot saved to {self.plot_path}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        print("Evaluation callback called with metrics:", metrics)  # Debug print
        if metrics is not None and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            self.eval_losses.append(eval_loss)
            self.eval_steps.append(self.current_step)
            print(f"Added evaluation point: step={self.current_step}, loss={eval_loss}")
            self.update_plot()

    def update_plot(self):
        if not self.losses:
            return

        method_title = ""
        if self.distillation_method:
            formatted_method = " ".join(
                word.capitalize() for word in self.distillation_method.split("_")
            )
            method_title = f" ({formatted_method})"

        colors = {
            "main": {
                "total_loss": "#1f77b4",  # Blue
                "kd_loss": "#ff7f0e",  # Orange
                "original_loss": "#2ca02c",  # Green
                "learning_rate": "#9467bd",  # Purple
                "eval_loss": "#d62728",  # Red
            },
            "avg": {
                "total_loss": "#7fb1d9",  # Light blue
                "kd_loss": "#ffb97f",  # Light orange
                "original_loss": "#7fcc7f",  # Light green
                "eval_loss": "#ff9896",  # Light red
            },
        }

        # Create figure with subplots
        fig_size = (10, 15) if len(self.eval_losses) > 0 else (10, 12)
        num_plots = 5 if len(self.eval_losses) > 0 else 4
        fig, axes = plt.subplots(num_plots, 1, figsize=fig_size)

        if num_plots == 5:
            ax1, ax2, ax3, ax4, ax5 = axes
        else:
            ax1, ax2, ax3, ax4 = axes

        fig.suptitle(f"Training Progress{method_title}", fontsize=16, fontweight="bold")

        # Plot total loss
        ax1.plot(
            self.steps,
            self.losses,
            color=colors["main"]["total_loss"],
            linestyle="-",
            linewidth=1.5,
        )
        self._add_moving_average(
            ax1, self.losses, self.steps, color=colors["avg"]["total_loss"]
        )
        ax1.set_title("Training Loss", fontweight="bold")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        self._add_epoch_lines(ax1)

        # Plot validation loss if available
        if len(self.eval_losses) > 0:
            ax1.plot(
                self.eval_steps,
                self.eval_losses,
                color=colors["main"]["eval_loss"],
                linestyle="-",
                linewidth=1.5,
                marker="o",
                markersize=4,
                label="Validation Loss",
            )
            ax1.legend(loc="upper right")

        # Plot knowledge distillation loss
        if any(x is not None for x in self.loss_kds):
            ax2.plot(
                self.steps,
                self.loss_kds,
                color=colors["main"]["kd_loss"],
                linestyle="-",
                linewidth=1.5,
            )
            self._add_moving_average(
                ax2, self.loss_kds, self.steps, color=colors["avg"]["kd_loss"]
            )
            ax2.set_title("Knowledge Distillation Loss", fontweight="bold")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Loss KD")
            ax2.grid(True, alpha=0.3)
            self._add_epoch_lines(ax2)

        # Plot original loss
        if any(x is not None for x in self.original_losses):
            ax3.plot(
                self.steps,
                self.original_losses,
                color=colors["main"]["original_loss"],
                linestyle="-",
                linewidth=1.5,
            )
            self._add_moving_average(
                ax3,
                self.original_losses,
                self.steps,
                color=colors["avg"]["original_loss"],
            )
            ax3.set_title("Student Cross-Entropy Loss", fontweight="bold")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Cross-Entropy Loss")
            ax3.grid(True, alpha=0.3)
            self._add_epoch_lines(ax3)

        # Plot learning rate
        if any(x is not None for x in self.learning_rates):
            ax4.plot(
                self.steps,
                self.learning_rates,
                color=colors["main"]["learning_rate"],
                linestyle="-",
                linewidth=1.5,
            )
            ax4.set_title("Learning Rate", fontweight="bold")
            ax4.set_xlabel("Step")
            ax4.set_ylabel("Learning Rate")
            ax4.grid(True, alpha=0.3)

        # Add a separate plot for validation loss if available
        if len(self.eval_losses) > 0:
            ax5.plot(
                self.eval_steps,
                self.eval_losses,
                color=colors["main"]["eval_loss"],
                linestyle="-",
                linewidth=1.5,
                marker="o",
                markersize=6,
            )
            ax5.set_title("Validation Loss", fontweight="bold")
            ax5.set_xlabel("Step")
            ax5.set_ylabel("Loss")
            ax5.grid(True, alpha=0.3)
            self._add_epoch_lines(ax5)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.plot_path)), exist_ok=True)

        # Save figure to file and close
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
        plt.savefig(self.plot_path)
        plt.close(fig)

    def _add_moving_average(self, ax, values, steps, color="#ff7f0e", **kwargs):
        """Helper method to add moving average to a plot."""
        # min num of points needed for moving average
        min_points = 5

        if len(values) > min_points:
            # filter out None values if present
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            if len(valid_indices) <= min_points:
                return

            valid_values = [values[i] for i in valid_indices]
            valid_steps = [steps[i] for i in valid_indices]

            # window size starts smaller, grow to max_window
            actual_window = min(
                self.moving_avg_window, max(min_points, len(valid_values) // 5)
            )

            cumsum = np.cumsum(np.insert(valid_values, 0, 0))
            avg_values = (
                cumsum[actual_window:] - cumsum[:-actual_window]
            ) / actual_window
            plot_steps = valid_steps[
                actual_window - 1 : len(avg_values) + actual_window - 1
            ]

            ax.plot(
                plot_steps,
                avg_values,
                color=color,
                linewidth=2,
                alpha=0.75,
                label=f"Moving Avg ({actual_window} steps)",
            )
            ax.legend(loc="upper right")

    def record_metrics(
        self,
        step,
        loss,
        loss_kd,
        original_loss,
        gradient_accumulation_steps,
        learning_rate,
        epoch,
    ):
        is_new_step = step != self.current_step

        if is_new_step and self.accumulation_count > 0 and self.current_step >= 0:
            # self.losses.append(loss)  # Use the trainer's accumulated loss
            alpha = self.config["distillation"]["alpha"]
            calculated_total = alpha * (
                self.accumulated_loss_kd / gradient_accumulation_steps
            ) + (1 - alpha) * (
                self.accumulated_original_loss / gradient_accumulation_steps
            )
            print(f"\n\nCalculated loss: {calculated_total}")
            print(f"Trainer's loss: {loss}\n\n")
            self.losses.append(calculated_total)
            self.steps.append(self.current_step)
            self.loss_kds.append(self.accumulated_loss_kd / gradient_accumulation_steps)
            self.original_losses.append(
                self.accumulated_original_loss / gradient_accumulation_steps
            )
            self.learning_rates.append(learning_rate)

            # if epoch has changed since last step
            current_epoch_int = int(epoch)
            if current_epoch_int != self.last_epoch and self.last_epoch != -1:
                self.epoch_end_steps.append(self.current_step)
                print(f"End of epoch {self.last_epoch} at step {self.current_step}")
            self.last_epoch = current_epoch_int

            # Update plot periodically
            if step > 0 and step % self.update_freq == 0:
                self.update_plot()
                # print(f"Updated plot at step {step}, saved to {self.plot_path}")

            # Reset accumulators
            self.accumulated_loss_kd = 0
            self.accumulated_original_loss = 0
            self.accumulation_count = 0

        # Update the current step
        self.current_step = step

        # Accumulate component losses
        self.accumulated_loss_kd += loss_kd
        self.accumulated_original_loss += original_loss
        self.accumulation_count += 1

    def _add_epoch_lines(self, ax):
        """Helper method to add vertical lines at epoch boundaries."""
        if not self.epoch_end_steps:
            return

        for i, step in enumerate(self.epoch_end_steps):
            # Add vertical line
            line = ax.axvline(
                x=step, color="red", linestyle="--", linewidth=1.2, alpha=0.7
            )

            # Add a text label for the epoch
            y_pos = ax.get_ylim()[1] * 0.95  # Position near the top
            ax.text(
                step + 5,
                y_pos,
                f"Epoch {i+1}",
                color="red",
                fontsize=8,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )


def get_max_token_length(
    dataset, tokenizer, generate_plot=True, plot_path="token_length_stats.png"
):
    max_length = 0
    lengths = []

   

    max_index = 1000
    for i, example in enumerate(dataset):
        try:
            # Format the example
            formatted_text = medlfqa_format(example)

            # Tokenize without padding or truncation to get true length
            tokens = tokenizer(formatted_text['text'], truncation=False, padding=False)

            # Get token count
            length = len(tokens.input_ids)
            lengths.append(length)

            # Update max length
            if length > max_length:
                max_length = length
                max_index = i
            # Print progress occasionally
            if i % 100 == 0:
                print(f"Processed {i} examples. Current max length: {max_length}")

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            print(f"Example keys: {list(example.keys())}")

    # Calculate statistics
    stats = {
        "max_length": max_length,
        "max_index": max_index,
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "min": np.min(lengths),
        "percentiles": {
            "25": np.percentile(lengths, 25),
            "50": np.percentile(lengths, 50),
            "75": np.percentile(lengths, 75),
            "90": np.percentile(lengths, 90),
            "95": np.percentile(lengths, 95),
            "99": np.percentile(lengths, 99),
        },
        "total_examples": len(lengths),
    }

    # Generate histogram plot
    if generate_plot:
        plt.figure(figsize=(12, 8))

        # Histogram
        plt.subplot(2, 1, 1)
        plt.hist(lengths, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            stats["mean"],
            color="red",
            linestyle="dashed",
            linewidth=1,
            label=f'Mean: {stats["mean"]:.1f}',
        )
        plt.axvline(
            stats["median"],
            color="green",
            linestyle="dashed",
            linewidth=1,
            label=f'Median: {stats["median"]:.1f}',
        )
        plt.axvline(
            stats["max_length"],
            color="purple",
            linestyle="dashed",
            linewidth=1,
            label=f'Max: {stats["max_length"]}',
        )
        plt.title("Token Length Distribution")
        plt.xlabel("Token Length")
        plt.ylabel("Number of Examples")
        plt.legend()
        plt.grid(alpha=0.3)

        # Box plot
        plt.subplot(2, 1, 2)
        plt.boxplot(lengths, vert=False, patch_artist=True)
        plt.title("Token Length Box Plot")
        plt.xlabel("Token Length")
        plt.grid(alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        stats["plot_path"] = plot_path

    print(f"Analysis complete. Maximum token length: {max_length}")
    print(f"Mean token length: {stats['mean']:.2f}")
    print(f"Median token length: {stats['median']:.2f}")
    print(f"95th percentile: {stats['percentiles']['95']:.2f}")

    return max_length, max_index, stats


if __name__ == "__main__":

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    config = load_config()
    dataset = (
        load_dataset(
            config["dataset"]["name"],
            config["dataset"]["subset"],
            split=config["dataset"]["split"],
        )
        if config["dataset"].get("subset")
        else load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    )
    
    dataset = dataset.map(medlfqa_format)
    print(dataset[0].keys()) # contains 'text'
    print(dataset[0]['text'])
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], token=HF_TOKEN
    )
    max_tokens, max_index, token_stats = get_max_token_length(
        dataset, student_tokenizer
    )
    print(f"Max token length: {max_tokens}, Max index: {max_index}")

    # Print detailed statistics
    print("\nToken Length Statistics:")
    print(f"Mean: {token_stats['mean']:.2f}")
    print(f"Median: {token_stats['median']:.2f}")
    print(f"Distribution:")
    for p, v in token_stats["percentiles"].items():
        print(f"  {p}th percentile: {v:.2f}")

    if "plot_path" in token_stats:
        print(f"\nToken length distribution plot saved to: {token_stats['plot_path']}")

    # print(f"\n\n{dataset[max_index]['question']}")
    # print(f"\n\n{dataset[max_index]['options']}")
    # print(f"\n\n{dataset[max_index]['model_response']}")
    # print(f"\n\n{dataset[max_index]['answer']}")
