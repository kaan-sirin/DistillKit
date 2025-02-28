import matplotlib.pyplot as plt
from transformers import TrainerCallback
import numpy as np
import os

# TODO: Title should reflects distillation mode
# TODO: Plot KLD and cross-entropy loss separately


class LivePlotCallback(TrainerCallback):
    def __init__(self, plot_path="./loss_plot.png", update_freq=20):
        self.losses = []
        self.learning_rates = []
        self.steps = []
        self.plot_path = plot_path
        self.update_freq = update_freq
        self.call_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:

            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)
            
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])
            
            # Update plot every `update_freq` steps
            self.call_count += 1
            if self.call_count % self.update_freq == 0:
                self.update_plot()
                print(f"Updated plot at step {state.global_step}, saved to {self.plot_path}")
                
    def on_train_end(self, args, state, control, **kwargs):
        self.update_plot()
        print(f"Final plot saved to {self.plot_path}")
    
    def update_plot(self):
        if not self.losses:
            return
            
        # New plot
        plt.figure(figsize=(10, 6))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Plot loss
        ax1.plot(self.steps, self.losses, 'b-')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot moving average
        if len(self.losses) > 10:
            window_size = min(10, len(self.losses) // 5)
            avg_losses = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            valid_steps = self.steps[window_size-1:][:len(avg_losses)]
            ax1.plot(valid_steps, avg_losses, 'r-', linewidth=2, alpha=0.7, label=f'Moving Avg ({window_size} steps)')
            ax1.legend()
        
        # Plot learning rate 
        if self.learning_rates:
            ax2.plot(self.steps, self.learning_rates, 'g-')
            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.plot_path)), exist_ok=True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close(fig)