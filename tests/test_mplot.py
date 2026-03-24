from nets.nn import Linear, Sequential
from nets.nn.activations import ReLU
from nets.visualization.model_viz import ModelVisualizer


model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 4),
    ReLU(),
    Linear(4, 1)
)

viz = ModelVisualizer(
    model,
    title="RaktimNet v1 🚀",
    color_map={
        "Linear": "#00C4E6",   # neon green
        "ReLU": "#BB00FF"      # bright red
    }
)

viz.show()