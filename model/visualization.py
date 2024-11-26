from graphviz import Digraph

# Initialize Digraph with a left-to-right layout
dot = Digraph(format="png", comment="Single Input LSTM Classifier")
dot.attr(rankdir='LR', splines='true', nodesep='1', ranksep='1', fontname='Helvetica', fontsize='10')

# Add input nodes for Prices in USD, Twitter followers, and development activity with Set2 colors
dot.node("Inputs", "Prices in USD", shape="ellipse", style="filled", fillcolor="#66C2A5", width='1.2')  # Light green
dot.node("input_twitter", "Twitter Followers", shape="ellipse", style="filled", fillcolor="#FC8D62", width='1.2')  # Coral
dot.node("input_dev", "Development Activity", shape="ellipse", style="filled", fillcolor="#8DA0CB", width='1.2')  # Light blue

# Add nodes for LSTM layers, dropout, ReLU, FC layer, and Sigmoid with Set2 colors
dot.node("lstm1", "LSTM1\n(input_dim → 128)", shape="box", style="rounded,filled", fillcolor="#66C2A5", width='2')  # Light green
dot.node("dropout1", "Dropout", shape="box", style="filled", fillcolor="#E78AC3", width='1.5')  # Light pink
dot.node("relu1", "ReLU", shape="box", style="filled", fillcolor="#8DA0CB", width='1.5')  # Light blue
dot.node("lstm2", "LSTM2\n(128 → 128)", shape="box", style="rounded,filled", fillcolor="#66C2A5", width='2')  # Light green
dot.node("dropout2", "Dropout", shape="box", style="filled", fillcolor="#E78AC3", width='1.5')  # Light pink
dot.node("relu2", "ReLU", shape="box", style="filled", fillcolor="#8DA0CB", width='1.5')  # Light blue
dot.node("fc", "Fully Connected\n(128 → 1)", shape="box", style="rounded,filled", fillcolor="#FC8D62", width='2')  # Coral
dot.node("sigmoid", "Sigmoid\n(Binary Output)", shape="ellipse", style="filled", fillcolor="#66C2A5", width='1.2')  # Light green

# Add edges to connect the inputs to the model
dot.edge("input_prices", "lstm1", label="Prices in USD", color="black")
dot.edge("input_twitter", "lstm1", label="Twitter Followers", color="black")
dot.edge("input_dev", "lstm1", label="Development Activity", color="black")

# Add edges between layers
dot.edge("lstm1", "dropout1", color="black")
dot.edge("dropout1", "relu1", color="black")
dot.edge("relu1", "lstm2", color="black")
dot.edge("lstm2", "dropout2", color="black")
dot.edge("dropout2", "relu2", color="black")
dot.edge("relu2", "fc", color="black")
dot.edge("fc", "sigmoid", label="Prediction", color="black")

# Save and render
dot.render("SingleInputLSTMClassifier_diagram_left_to_right_128_pretty_Set2", view=True)
