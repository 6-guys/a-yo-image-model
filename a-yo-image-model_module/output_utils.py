import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np


def show_result(frame_0, generated_frames, n=10):

    plt.figure(figsize=(20, 4))
    for i in range(n):
        if i == 0 :
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(frame_0[i].reshape(128, 128, 4), cmap="gray")
            plt.axis("off")
        else:
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(np.clip(generated_frames[i],0, 1).reshape(128, 128, 4), cmap="gray")
            plt.axis("off")
    plt.show()

last_frame_num = -1

def animate_frames(frame_0, predicted_frames):
    """
    Create and display an animation from predicted frames

    Parameters:
    predicted_frames (numpy array): Array of predicted frames, expected shape (num_frames, 128, 128, 3).
    """
    num_frames = predicted_frames.shape[0]

    fig, ax = plt.subplots()
    img = ax.imshow(np.clip(frame_0, 0, 1).reshape(128, 128, 4), animated=True)
    ax.axis('off') 
    

    def update(frame_num):
        global last_frame_num
        if frame_num == last_frame_num:
            return [img]
        
        last_frame_num = frame_num
        if frame_num == 0:
            img.set_array(np.clip(frame_0, 0, 1).reshape(128, 128, 4))
        else:
            img.set_array(np.clip(predicted_frames[frame_num], 0, 1).reshape(128, 128, 4)) 
        print(frame_num)
        return [img]

    ani = FuncAnimation(
        fig, update, frames=num_frames, interval=100, blit=False
    )

    plt.show()