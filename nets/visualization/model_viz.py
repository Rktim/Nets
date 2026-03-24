import plotly.graph_objects as go


class ModelVisualizer:

    def __init__(self, model, title="Interactive Model Architecture", color_map=None):

        self.model = model
        self.title = title

        # default color theme
        self.default_colors = {
            "Linear": "#4FC3F7",
            "ReLU": "#FFB74D",
            "Sigmoid": "#81C784",
            "Tanh": "#BA68C8",
            "Dropout": "#E57373"
        }

        # override with user colors if provided
        if color_map:
            self.default_colors.update(color_map)

    # -------------------------
    # COLOR RESOLUTION
    # -------------------------
    def get_color(self, layer):

        name = layer.__class__.__name__

        for key in self.default_colors:
            if key in name:
                return self.default_colors[key]

        return "#B0BEC5"  # fallback gray

    # -------------------------
    # LAYER EXTRACTION
    # -------------------------
    def extract_layers(self):

        if hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            layers = [self.model]

        nodes = []

        for i, layer in enumerate(layers):

            name = layer.__class__.__name__
            color = self.get_color(layer)

            # shape + params
            if hasattr(layer, "weight"):
                in_f = layer.weight.data.shape[0]
                out_f = layer.weight.data.shape[1]

                label = f"{name}<br>{in_f} → {out_f}"
                params = in_f * out_f + out_f
            else:
                label = name
                params = 0

            nodes.append({
                "id": i,
                "label": label,
                "color": color,
                "params": params
            })

        return nodes

    # -------------------------
    # GRAPH BUILD
    # -------------------------
    def build_graph(self):

        nodes = self.extract_layers()

        x = [i for i in range(len(nodes))]
        y = [0] * len(nodes)

        labels = [n["label"] for n in nodes]
        colors = [n["color"] for n in nodes]

        hover = [
            f"{n['label']}<br>Params: {n['params']}"
            for n in nodes
        ]

        edge_x = []
        edge_y = []

        for i in range(len(nodes) - 1):
            edge_x += [i, i + 1, None]
            edge_y += [0, 0, None]

        return x, y, labels, colors, hover, edge_x, edge_y

    # -------------------------
    # SHOW
    # -------------------------
    def show(self):

        x, y, labels, colors, hover, edge_x, edge_y = self.build_graph()

        fig = go.Figure()

        # edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2, color='black'),
            hoverinfo='none'
        ))

        # nodes
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            marker=dict(size=55, color=colors),
            text=labels,
            textposition="bottom center",
            hovertext=hover,
            hoverinfo="text"
        ))

        fig.update_layout(
            title=self.title,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white"
        )

        fig.show()