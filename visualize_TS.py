import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load the ETTh1 dataset
def load_etth1():
    filepath = "raw_data/female_births_dataset.csv"
    df = pd.read_csv(filepath)

    # Convert the first column to datetime and set it as the index
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)

    return df

# Plot the selected column
def plot_column(df, column_name):
    fig, ax = plt.subplots()
    ax.plot(df.index, df[column_name], label=column_name)
    ax.set_title(f"Column: {column_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    # Enable zooming and panning
    plt.tight_layout()
    plt.show()

# Plot all columns
def plot_all_columns(df):
    fig, ax = plt.subplots()
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)
    ax.set_title("All Columns in ETTh1 Dataset")
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend()

    # Enable zooming and panning
    plt.tight_layout()
    plt.show()

def main():
    df = load_etth1()

    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    # Ask the user whether to plot all columns or a specific one
    choice = input("Do you want to plot all columns? (yes/no): ").strip().lower()
    if choice == "yes":
        plot_all_columns(df)
    else:
        while True:
            try:
                col_index = int(input("Enter the column number to plot: "))
                if 0 <= col_index < len(df.columns):
                    selected_column = df.columns[col_index]
                    plot_column(df, selected_column)
                else:
                    print("Invalid column number. Try again.")
            except ValueError:
                print("Please enter a valid number.")

if __name__ == "__main__":
    main()
