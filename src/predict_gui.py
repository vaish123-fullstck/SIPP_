import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/synthetic_dataset_with_districts.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/impact_model.pkl")

# === Load dataset and model ===
try:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error: {e}")
    exit()

# === Predict and Display ===
def predict_impact():
    selected_district = district_var.get()
    if not selected_district:
        messagebox.showerror("Input Error", "Please select a district.")
        return

    district_data = df[df["District"] == selected_district]

    if district_data.empty:
        messagebox.showerror("Data Error", f"No data found for {selected_district}.")
        return

    features = ['Budget', 'Target_Audience', 'Location', 'Sustainability_Factors']
    X = district_data[features].values
    predictions = model.predict(X)
    avg_score = np.mean(predictions)

    # Display score
    result_text.set(f"Predicted Impact Score for {selected_district}: {avg_score:.2f}")
    display_pros_cons(avg_score)
    show_graph(avg_score)

def display_pros_cons(score):
    if score > 70:
        pros = "✔ High sustainability, good public acceptance, strong economic benefits."
        cons = "⚠ High initial investment may be needed."
    elif score > 40:
        pros = "✔ Balanced benefits, moderate success expected."
        cons = "⚠ Some sustainability concerns or policy hurdles."
    else:
        pros = "✔ Cost-effective in short term."
        cons = "⚠ Low impact, sustainability and public interest issues."

    pros_text.set(pros)
    cons_text.set(cons)

def show_graph(score):
    fig = plt.Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(["Impact Score"], [score], color="teal")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Impact Score")
    ax.set_title("District Project Impact")

    # Clear previous if any
    for widget in graph_frame.winfo_children():
        widget.destroy()

    chart = FigureCanvasTkAgg(fig, graph_frame)
    chart.get_tk_widget().pack()

# === GUI ===
root = tk.Tk()
root.title("SIPP - Social Impact Prediction Platform")
root.geometry("600x500")
root.configure(bg="#f5f5f5")

# === Header ===
tk.Label(root, text="Select a District", font=("Arial", 14), bg="#f5f5f5").pack(pady=10)

# === Dropdown ===
district_var = tk.StringVar()
districts = sorted(df["District"].unique())
dropdown = ttk.Combobox(root, textvariable=district_var, values=districts, state="readonly", width=30)
dropdown.pack()

# === Predict Button ===
tk.Button(root, text="Predict Impact", command=predict_impact, bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5).pack(pady=15)

# === Result Display ===
result_text = tk.StringVar()
tk.Label(root, textvariable=result_text, font=("Arial", 12), bg="#f5f5f5", fg="#333").pack()

# === Graph Frame ===
graph_frame = tk.Frame(root, bg="#f5f5f5")
graph_frame.pack(pady=10)

# === Pros and Cons Display ===
pros_text = tk.StringVar()
cons_text = tk.StringVar()

tk.Label(root, text="Pros:", font=("Arial", 12, "bold"), bg="#f5f5f5", fg="green").pack()
tk.Label(root, textvariable=pros_text, font=("Arial", 11), bg="#f5f5f5", wraplength=500, justify="center").pack(pady=2)

tk.Label(root, text="Cons:", font=("Arial", 12, "bold"), bg="#f5f5f5", fg="red").pack()
tk.Label(root, textvariable=cons_text, font=("Arial", 11), bg="#f5f5f5", wraplength=500, justify="center").pack(pady=2)

# === Run ===
root.mainloop()
