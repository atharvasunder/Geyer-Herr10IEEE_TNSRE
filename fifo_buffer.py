# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:42:13 2025

@author: athar
"""

from collections import deque

class FIFOBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)  # Automatically removes oldest item when full

    def push(self, item):
        """Add an item to the buffer and return the removed item if full."""
        removed_item = 0
        if len(self.buffer) == self.size:  # Buffer is full
            removed_item = self.buffer[0]  # Get the oldest item before removing
        self.buffer.append(item)  # Append the new item (automatically removes oldest if full)
        return removed_item  # Return the removed item (or None if not full)

    def pop(self):
        """Remove and return the oldest item (FIFO order)."""
        if self.buffer:
            return self.buffer.popleft()
        raise IndexError("Buffer is empty")

    def peek(self):
        """View the oldest item without removing it."""
        if self.buffer:
            return self.buffer[0]
        return 0

    def is_full(self):
        """Check if buffer is full."""
        return len(self.buffer) == self.size

    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.buffer) == 0

    def __repr__(self):
        return f"FIFOBuffer({list(self.buffer)})"

# Example Usage
# fifo = FIFOBuffer(5)
# fifo.push(10)
# fifo.push(20)
# fifo.push(30)
# print(fifo)  # FIFOBuffer([10, 20, 30])
# print(fifo.pop())  # 10 (first item pushed)
# print(fifo)  # FIFOBuffer([20, 30])
