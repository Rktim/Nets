# from dash import Dash, dcc, html, dash_table
# from dash.dependencies import Input, Output
# import dash
# import plotly.graph_objs as go
# import numpy as np

# from nets.visualization.utils import ema


# THEME = {
#     "bg": "#0B1220",
#     "panel": "#111827",
#     "text": "#E5E7EB",
#     "subtext": "#9CA3AF",
#     "loss": "#EF4444",
#     "acc": "#10B981",
# }

# COLORS = [
#     "#3B82F6", "#EF4444", "#10B981",
#     "#F59E0B", "#8B5CF6", "#14B8A6"
# ]


# class Dashboard:

#     def __init__(self, logger):

#         self.logger = logger
#         self.app = Dash(__name__)

#         self.app.layout = html.Div([

#             html.H1("🚀 Nets ML Platform", style={"color": THEME["text"], "textAlign": "center"}),

#             html.Div([

#                 # -------- SIDEBAR --------
#                 html.Div([

#                     html.H3("Experiments", style={"color": THEME["text"]}),

#                     dcc.Dropdown(id="run-selector", multi=True),

#                     html.Button("🗑 Delete Selected", id="delete-btn", n_clicks=0, style=btn("#EF4444")),
#                     html.Button("⚠ Clear All", id="clear-btn", n_clicks=0, style=btn("#B91C1C")),

#                     html.Hr(),

#                     html.Label("Filter Model", style=label()),
#                     dcc.Dropdown(id="filter-model", multi=True),

#                     html.Label("Filter Dataset", style=label()),
#                     dcc.Dropdown(id="filter-dataset", multi=True),

#                     html.Div(id="meta-info")

#                 ], style=sidebar()),

#                 # -------- MAIN --------
#                 html.Div([

#                     # -------- TABLE
#                     html.Div([
#                         html.H3("Runs", style={"color": THEME["text"]}),

#                         dash_table.DataTable(
#                             id="runs-table",
#                             row_selectable="multi",
#                             sort_action="native",
#                             style_header=table_header(),
#                             style_cell=table_cell(),
#                         )
#                     ], style=section()),

#                     # -------- CARDS
#                     html.Div(id="cards", style=section()),

#                     # -------- LOSS
#                     html.Div([dcc.Graph(id="loss")], style=section()),

#                     # -------- ACC
#                     html.Div([dcc.Graph(id="acc")], style=section()),

#                 ], style={"width": "80%", "padding": "20px"})

#             ], style={"display": "flex", "gap": "20px"}),

#             dcc.Interval(id="interval", interval=2000)

#         ], style={"backgroundColor": THEME["bg"], "padding": "20px"})

#         # ---------------- CALLBACK ----------------

#         @self.app.callback(
#             [
#                 Output("run-selector", "options"),
#                 Output("filter-model", "options"),
#                 Output("filter-dataset", "options"),
#                 Output("runs-table", "data"),
#                 Output("runs-table", "columns"),
#                 Output("cards", "children"),
#                 Output("loss", "figure"),
#                 Output("acc", "figure"),
#             ],
#             [
#                 Input("interval", "n_intervals"),
#                 Input("run-selector", "value"),
#                 Input("filter-model", "value"),
#                 Input("filter-dataset", "value"),
#                 Input("runs-table", "selected_rows"),
#                 Input("delete-btn", "n_clicks"),
#                 Input("clear-btn", "n_clicks"),
#             ]
#         )
#         def update(n, selected_runs, model_filter, dataset_filter,
#                    table_selected, delete_clicks, clear_clicks):

#             ctx = dash.callback_context

#             runs = self.logger.get_runs()

#             # -------- DELETE HANDLING
#             if ctx.triggered:
#                 trigger = ctx.triggered[0]["prop_id"].split(".")[0]

#                 if trigger == "delete-btn" and selected_runs:
#                     self.logger.delete_runs(selected_runs)

#                 if trigger == "clear-btn":
#                     self.logger.clear_all()

#                 runs = self.logger.get_runs()

#             if not runs:
#                 return [], [], [], [], [], [], {}, {}

#             # -------- FILTER
#             filtered = {}
#             for rid, r in runs.items():
#                 if model_filter and r["meta"].get("model") not in model_filter:
#                     continue
#                 if dataset_filter and r["meta"].get("dataset") not in dataset_filter:
#                     continue
#                 filtered[rid] = r

#             options = [{"label": r["name"], "value": rid} for rid, r in filtered.items()]

#             # -------- TABLE DATA
#             table_data = []
#             for rid, r in filtered.items():
#                 row = {
#                     "id": rid,
#                     "name": r["name"],
#                     "model": r["meta"].get("model"),
#                     "dataset": r["meta"].get("dataset"),
#                 }
#                 row.update(r["meta"].get("hyperparams", {}))
#                 table_data.append(row)

#             columns = [{"name": k, "id": k} for k in table_data[0].keys()] if table_data else []

#             # -------- SYNC TABLE
#             if table_selected:
#                 selected_runs = [table_data[i]["id"] for i in table_selected]

#             if not selected_runs:
#                 selected_runs = list(filtered.keys())[:2]

#             # -------- CARDS
#             last_run = runs[selected_runs[-1]]
#             logs = [l for l in last_run["logs"] if l["type"] == "train"]

#             cards = html.Div()
#             if logs:
#                 last = logs[-1]
#                 cards = html.Div([
#                     card("Loss", last["loss"]),
#                     card("Accuracy", last["accuracy"]),
#                     card("Epoch", last["epoch"]),
#                 ], style=grid())

#             # -------- LOSS
#             loss_fig = go.Figure()
#             for i, rid in enumerate(selected_runs):
#                 logs = [l for l in runs[rid]["logs"] if l["type"] == "train"]
#                 loss_fig.add_trace(go.Scatter(
#                     x=[l["epoch"] for l in logs],
#                     y=ema([l["loss"] for l in logs]),
#                     name=runs[rid]["name"],
#                     line=dict(color=COLORS[i % len(COLORS)], width=3)
#                 ))
#             style_fig(loss_fig, "Loss Comparison")

#             # -------- ACC
#             acc_fig = go.Figure()
#             for i, rid in enumerate(selected_runs):
#                 logs = [l for l in runs[rid]["logs"] if l["type"] == "train"]
#                 acc_fig.add_trace(go.Scatter(
#                     x=[l["epoch"] for l in logs],
#                     y=ema([l["accuracy"] for l in logs]),
#                     name=runs[rid]["name"],
#                     line=dict(color=COLORS[i % len(COLORS)], width=3)
#                 ))
#             style_fig(acc_fig, "Accuracy Comparison")

#             return (
#                 options,
#                 make_opts(runs, "model"),
#                 make_opts(runs, "dataset"),
#                 table_data,
#                 columns,
#                 cards,
#                 loss_fig,
#                 acc_fig
#             )

#     def run(self):
#         self.app.run(debug=False)


# # ---------------- STYLE HELPERS ----------------

# def section():
#     return {
#         "background": THEME["panel"],
#         "padding": "20px",
#         "borderRadius": "12px",
#         "marginBottom": "20px"
#     }

# def sidebar():
#     return {
#         "width": "20%",
#         "background": THEME["panel"],
#         "padding": "20px",
#         "borderRadius": "12px"
#     }

# def label():
#     return {"color": THEME["subtext"]}

# def btn(color):
#     return {
#         "marginTop": "10px",
#         "background": color,
#         "color": "white",
#         "border": "none",
#         "padding": "10px",
#         "borderRadius": "8px",
#         "cursor": "pointer"
#     }

# def table_header():
#     return {"backgroundColor": THEME["panel"], "color": THEME["text"]}

# def table_cell():
#     return {
#         "backgroundColor": THEME["bg"],
#         "color": THEME["text"],
#         "border": "1px solid #1f2937"
#     }

# def card(title, value):
#     return html.Div([
#         html.P(title, style={"color": THEME["subtext"]}),
#         html.H2(f"{value:.4f}" if isinstance(value, float) else str(value),
#                 style={"color": THEME["text"]})
#     ], style=section())

# def grid():
#     return {"display": "grid", "gridTemplateColumns": "repeat(3,1fr)", "gap": "20px"}

# def style_fig(fig, title):
#     fig.update_layout(
#         title=title,
#         template="plotly_dark",
#         paper_bgcolor=THEME["panel"],
#         plot_bgcolor=THEME["panel"],
#         font=dict(color=THEME["text"])
#     )

# def make_opts(runs, key):
#     return [{"label": v["meta"].get(key), "value": v["meta"].get(key)}
#             for v in runs.values()]

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np

# ---------- THEME ----------
BG = "#0b0f1a"
CARD = "#111827"
TEXT = "#E5E7EB"
GRID = "#1F2937"

COLORS = {
    "loss": "#FF4D4D",
    "acc": "#22C55E",
    "precision": "#38BDF8",
    "recall": "#F59E0B",
    "f1": "#A78BFA"
}


def fig_layout(title):
    return dict(
        template="plotly_dark",
        title=title,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID)
    )


def card(title, value):
    return html.Div([
        html.Div(title, style={"fontSize": "14px", "color": "#9CA3AF"}),
        html.Div(str(round(value, 4)), style={
            "fontSize": "26px", "fontWeight": "bold"
        })
    ], style={
        "background": CARD,
        "padding": "15px",
        "borderRadius": "10px"
    })


class Dashboard:

    def __init__(self, logger):
        self.logger = logger
        self.app = Dash(__name__)

        self.app.layout = html.Div([

            html.H2("🚀 Nets ML Dashboard", style={"textAlign": "center"}),

            # ---------- SIDEBAR + MAIN ----------
            html.Div([

                # SIDEBAR
                html.Div([
                    html.H4("Experiments"),
                    dcc.Dropdown(id="runs", multi=True),

                    html.Br(),
                    html.Button("Delete Selected", id="delete-btn")
                ], style={
                    "width": "20%",
                    "padding": "15px",
                    "background": CARD,
                    "borderRadius": "10px"
                }),

                # MAIN
                html.Div([

                    html.Div(id="cards", style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3,1fr)",
                        "gap": "10px"
                    }),

                    html.Div([
                        dcc.Graph(id="loss"),
                        dcc.Graph(id="acc")
                    ], style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "10px"
                    }),

                    html.Div(id="extra-metrics"),

                    dcc.Graph(id="comparison"),

                    dcc.Graph(id="confusion")

                ], style={"width": "80%", "padding": "15px"})

            ], style={"display": "flex", "gap": "15px"}),

            dcc.Interval(id="interval", interval=2000)

        ], style={"backgroundColor": BG, "color": TEXT, "padding": "20px"})

        # ---------- CALLBACK ----------
        @self.app.callback(
            Output("runs", "options"),
            Output("runs", "value"),
            Output("cards", "children"),
            Output("loss", "figure"),
            Output("acc", "figure"),
            Output("extra-metrics", "children"),
            Output("comparison", "figure"),
            Output("confusion", "figure"),
            Input("interval", "n_intervals"),
            Input("delete-btn", "n_clicks"),
            Input("runs", "value")
        )
        def update(_, delete_clicks, selected):

            runs = self.logger.get_runs()
            if not runs:
                return [], [], [], {}, {}, "", {}, {}

            options = [{"label": r["name"], "value": rid}
                       for rid, r in runs.items()]

            selected = selected or list(runs.keys())[-1:]

            # DELETE
            if delete_clicks:
                for rid in selected:
                    self.logger.delete_run(rid)
                return options, [], [], {}, {}, "", {}, {}

            run = runs[selected[-1]]
            logs = run["logs"]
            meta = run.get("meta", {})

            task = meta.get("task", "classification")

            # ---------- METRICS ----------
            epochs = [l["epoch"] for l in logs if "epoch" in l]
            loss = [l.get("loss") for l in logs if "loss" in l]
            acc = [l.get("accuracy") for l in logs if "accuracy" in l]

            precision = [l.get("precision") for l in logs if "precision" in l]
            recall = [l.get("recall") for l in logs if "recall" in l]
            f1 = [l.get("f1") for l in logs if "f1" in l]

            # ---------- CARDS ----------
            cards_ui = []
            if loss:
                cards_ui.append(card("Loss", loss[-1]))
            if acc:
                cards_ui.append(card("Accuracy", acc[-1]))
            if epochs:
                cards_ui.append(card("Epoch", epochs[-1]))

            # ---------- LOSS ----------
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=epochs[:len(loss)], y=loss,
                name=run["name"], line=dict(color=COLORS["loss"])
            ))
            loss_fig.update_layout(**fig_layout("Loss"))

            # ---------- ACC ----------
            acc_fig = go.Figure()
            acc_fig.add_trace(go.Scatter(
                x=epochs[:len(acc)], y=acc,
                name=run["name"], line=dict(color=COLORS["acc"])
            ))
            acc_fig.update_layout(**fig_layout("Accuracy"))

            # ---------- EXTRA (AUTO TASK) ----------
            extra = []

            if task == "classification":
                fig = go.Figure()

                if precision:
                    fig.add_trace(go.Scatter(
                        x=epochs[:len(precision)], y=precision,
                        name="Precision", line=dict(color=COLORS["precision"])
                    ))
                if recall:
                    fig.add_trace(go.Scatter(
                        x=epochs[:len(recall)], y=recall,
                        name="Recall", line=dict(color=COLORS["recall"])
                    ))
                if f1:
                    fig.add_trace(go.Scatter(
                        x=epochs[:len(f1)], y=f1,
                        name="F1", line=dict(color=COLORS["f1"])
                    ))

                fig.update_layout(**fig_layout("Classification Metrics"))
                extra = dcc.Graph(figure=fig)

            # ---------- COMPARISON ----------
            comp = go.Figure()

            for rid in selected:
                r = runs[rid]
                tl = [l for l in r["logs"] if "loss" in l]

                comp.add_trace(go.Scatter(
                    x=[l["epoch"] for l in tl],
                    y=[l["loss"] for l in tl],
                    name=r["name"]
                ))

            comp.update_layout(**fig_layout("Run Comparison"))

            # ---------- CONFUSION ----------
            cm_fig = go.Figure()

            cm_logs = [l for l in logs if l.get("type") == "confusion"]
            if cm_logs:
                cm = np.array(cm_logs[-1]["cm"])
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, colorscale="Turbo"
                ))

            cm_fig.update_layout(**fig_layout("Confusion Matrix"))

            return options, selected, cards_ui, loss_fig, acc_fig, extra, comp, cm_fig

    def run(self):
        self.app.run(debug=False)