import numpy as np
from abc import ABC, abstractmethod

class BaseFeedback(ABC):
    """
    Abstract base class for the feedback/nervous system components of the neuromuscular controller.
    """

    def __init__(self):
        """
        Initializes the abstract base class.
        """
        pass

    @abstractmethod
    def compute_stimulation(self, gait_phase):
        """
        Computes and returns the updated muscle stimulation based on the feedback pathway.

        Args:
            gait_phase (str): 'stance' or 'swing'

        Returns:
            float: The updated muscle stimulation.
        """
        pass

    @abstractmethod
    def update_buffer(self, force, length):
        """
        Updates the internal feedback buffers with the given force and length.

        Args:
            force: Current muscle-tendon complex force.
            length: Current length of the muscle-tendon complex.
        """
        pass

    @abstractmethod
    def reset(self, l_ce):
        """
        Resets and initializes the feedback buffers to their starting state.

        Should be called when entering stance phase to prepare the buffers
        with appropriate initial values and lengths based on delay constants.
        """
        pass
