import matplotlib.pyplot as plt
from transformers import TrainerCallback
import numpy as np
import os

# TODO: Add vertical red lines at the end of each epoch

class LivePlotCallback(TrainerCallback):
    def __init__(
        self, plot_path="./loss_plot.png", update_freq=20, moving_avg_window=10, distillation_method=None
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


        # For accumulating losses over gradient accumulation steps
        self.accumulated_loss_kd = 0
        self.accumulated_original_loss = 0
        self.accumulation_count = 0
        self.current_step = -1
        
        self.epoch_end_steps = []
        self.last_epoch = -1

    def on_train_end(self, args, state, control, **kwargs):
        if self.accumulation_count > 0:
            self.losses.append(state.log_history[-1].get("loss") if state.log_history else None)
            self.steps.append(self.current_step)
            self.loss_kds.append(self.accumulated_loss_kd)
            self.original_losses.append(self.accumulated_original_loss)
            self.learning_rates.append(self.learning_rates[-1] if self.learning_rates else 0)

        self.update_plot()
        print(f"Final plot saved to {self.plot_path}")

    def update_plot(self):
        if not self.losses:
            return
                
        method_title = ""
        if self.distillation_method:
            formatted_method = " ".join(word.capitalize() for word in self.distillation_method.split("_"))
            method_title = f" ({formatted_method})"

        colors = {
            "main": {
                "total_loss": "#1f77b4",  # Blue
                "kd_loss": "#ff7f0e",  # Orange
                "original_loss": "#2ca02c",  # Green
                "learning_rate": "#9467bd",  # Purple
            },
            "avg": {
                "total_loss": "#7fb1d9",  # Light blue
                "kd_loss": "#ffb97f",  # Light orange
                "original_loss": "#7fcc7f",  # Light green
            },
        }

        # Create new plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))

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

    def record_metrics(self, step, loss, loss_kd, original_loss,gradient_accumulation_steps, learning_rate, epoch):
        is_new_step = step != self.current_step

        if is_new_step and self.accumulation_count > 0 and self.current_step >= 0:
            self.losses.append(loss)  # Use the trainer's accumulated loss
            self.steps.append(self.current_step)
            self.loss_kds.append(self.accumulated_loss_kd / gradient_accumulation_steps)
            self.original_losses.append(self.accumulated_original_loss / gradient_accumulation_steps)
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
            line = ax.axvline(x=step, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            
            # Add a text label for the epoch
            y_pos = ax.get_ylim()[1] * 0.95  # Position near the top
            ax.text(step + 5, y_pos, f"Epoch {i+1}", 
                    color='red', fontsize=8, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
