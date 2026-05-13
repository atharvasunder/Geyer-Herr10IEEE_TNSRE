import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class WalkVisualizer:
    def __init__(self, rigid_body_system):
        self.rbs = rigid_body_system
        self.fig = None

    def animate(self, indices, interval=33, save_path=None):
        """
        Uses Matplotlib's built-in FuncAnimation to create a robust and
        flicker-free animation without struggling with raw GUI loops.

        Args:
            indices: frame indices to render
            interval: ms between frames
            save_path: if provided, save the animation as an MP4 to this path
        """
        self.fig, ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.manager.set_window_title("Rigid Body Animation")
        
        # ax.set_title("X-Z View")
        ax.set_aspect('equal')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.1, 2)
        ax.grid(True)
        
        # Ground Line
        ax.plot([-1000, 1000], [0, 0], color='gray', linewidth=2, zorder=1)
        
        # Prepare line artists for each body link — color by segment
        body_lines = []
        for body in self.rbs.body_list:
            if 'trunk' in body.name:
                color = 'green'
            elif body.name.endswith('_1'):
                color = 'red'
            else:
                color = 'blue'
            line, = ax.plot([], [], color=color, linewidth=4, solid_capstyle='round', zorder=2)
            body_lines.append(line)
        
        def init():
            for line in body_lines:
                line.set_data([], [])
            return body_lines

        def update(frame_idx):
            idx = indices[frame_idx]
            for j, body in enumerate(self.rbs.body_list):
                x_w, z_w = body.update_body_geometry(
                    body.x_list[idx], body.z_list[idx], body.p_list[idx]
                )
                body_lines[j].set_data(x_w, z_w)
                
            return body_lines

        # blit=True ONLY updates the changing data, removing the weird "vibrations" and stutter
        self.anim = animation.FuncAnimation(
            self.fig, 
            update, 
            frames=len(indices),
            init_func=init, 
            interval=interval, 
            blit=True, 
            repeat=False
        )

        # Save the animation as MP4 if a path is provided
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fps = max(1, int(1000 / interval))

            # Use ffmpeg binary bundled with imageio-ffmpeg
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

            print(f"Saving animation to {save_path} ...")
            self.anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved: {save_path}")
        
        # This acts as a blocking call during the animation.
        # It handles all flush_events safely internally.
        plt.show()
