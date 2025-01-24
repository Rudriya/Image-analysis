import tkinter as tk
from tkinter import  filedialog, messagebox, ttk
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.patches import Circle, Rectangle
import pandas as pd
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
from matplotlib.lines import Line2D
from shapely.geometry import Polygon as ShapelyPolygon
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from skimage import measure


def preprocess_data(file_path):
    """
    Preprocess data from the provided Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        numpy.ndarray: Cleaned data as a NumPy array.
    """
    try:
        # Load and clean the data
        data = pd.read_excel(file_path, header=None).dropna(how='all', axis=0).dropna(how='all', axis=1)
        numeric_data = data.apply(pd.to_numeric, errors='coerce').fillna(0)  # Replace non-numeric values with 0
        return numeric_data.to_numpy()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file: {e}")
        return None

def trim_to_common_shape(data1, data2):
    """
    Trim two datasets to the same shape.

    Args:
        data1 (numpy.ndarray): First dataset.
        data2 (numpy.ndarray): Second dataset.

    Returns:
        tuple: Trimmed datasets.
    """
    min_rows = min(data1.shape[0], data2.shape[0])
    min_cols = min(data1.shape[1], data2.shape[1])
    return data1[:min_rows, :min_cols], data2[:min_rows, :min_cols]

def perform_operation(data1, data2, operation):
    """
    Perform the selected arithmetic operation on the data.

    Args:
        data1 (numpy.ndarray): First dataset.
        data2 (numpy.ndarray): Second dataset.
        operation (str): Operation to perform ("Add", "Subtract", "Multiply", "Divide").

    Returns:
        numpy.ndarray: Resulting dataset.
    """
    try:
        if operation == "Add":
            return np.add(data1, data2)
        elif operation == "Subtract":
            return np.subtract(data1, data2)
        elif operation == "Multiply":
            return np.multiply(data1, data2)
        elif operation == "Divide":
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(data1, data2)
                result[np.isnan(result)] = 0  # Replace NaNs with 0
                return result
    except Exception as e:
        messagebox.showerror("Error", f"Operation failed: {e}")
        return None


selected_shape = None

def analyze_region(data, shape, coords):
    """
    Analyzes the selected region of the image data.

    Args:
        data (numpy.ndarray): The cleaned temperature data.
        shape (str): The type of shape ('line', 'rectangle', 'circle').
        coords (list): The coordinates for the shape.
                       - Line: [(x1, y1), (x2, y2)]
                       - Rectangle: [(x1, y1), (x2, y2)]
                       - Circle: [(x_center, y_center), (radius, None)]

    Returns:
        dict: Statistics including average, min, max in Celsius and Kelvin,
              and the area.
    """
    if shape == "line":
        x1, y1 = coords[0]
        x2, y2 = coords[1]
    
    # Generate evenly spaced points along the line
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1  # Number of points to interpolate
        x, y = np.linspace(x1, x2, num_points), np.linspace(y1, y2, num_points)
        indices = np.round([x, y]).astype(int)  # Convert to integer indices

    # Ensure indices are within the bounds of the data array
        indices = np.clip(indices, 0, np.array(data.shape)[:, None] - 1)

    # Extract data values along the line
        region_data = data[indices[1], indices[0]]

    # Remove NaN values
        region_data = region_data[~np.isnan(region_data)]

        if region_data.size == 0:
            return {
            "average_celsius": None, "average_kelvin": None,
            "minimum_celsius": None, "minimum_kelvin": None,
            "maximum_celsius": None, "maximum_kelvin": None,
            "area": 0
            }

    # Calculate statistics
        avg_temp_celsius = np.mean(region_data)
        min_temp_celsius = np.min(region_data)
        max_temp_celsius = np.max(region_data)

        return {
        "average_celsius": avg_temp_celsius,
        "average_kelvin": avg_temp_celsius + 273.15,
        "minimum_celsius": min_temp_celsius,
        "minimum_kelvin": min_temp_celsius + 273.15,
        "maximum_celsius": max_temp_celsius,
        "maximum_kelvin": max_temp_celsius + 273.15,
        "area": len(region_data)
        }


    elif shape == "rectangle":
        x1, y1 = coords[0]
        x2, y2 = coords[1]

    # Ensure coordinates are within bounds
        x_min, x_max = sorted([max(0, x1), min(data.shape[1] - 1, x2)])
        y_min, y_max = sorted([max(0, y1), min(data.shape[0] - 1, y2)])

    # Extract the rectangular region
        region_data = data[y_min:y_max+1, x_min:x_max+1].flatten()

    # Remove NaN values
        region_data = region_data[~np.isnan(region_data)]

        if region_data.size == 0:
            return {
            "average_celsius": None, "average_kelvin": None,
            "minimum_celsius": None, "minimum_kelvin": None,
            "maximum_celsius": None, "maximum_kelvin": None,
            "area": 0
            }

    # Calculate statistics
        avg_temp_celsius = np.mean(region_data)
        min_temp_celsius = np.min(region_data)
        max_temp_celsius = np.max(region_data)

        return {
        "average_celsius": avg_temp_celsius,
        "average_kelvin": avg_temp_celsius + 273.15,
        "minimum_celsius": min_temp_celsius,
        "minimum_kelvin": min_temp_celsius + 273.15,
        "maximum_celsius": max_temp_celsius,
        "maximum_kelvin": max_temp_celsius + 273.15,
        "area": len(region_data)
        }


  
    elif shape == "circle":
       if len(coords) != 2 or not isinstance(coords[1], int):
            print("Invalid circle coordinates provided. Ensure the center and radius are correctly defined.")
            return {
            "average_celsius": None, "average_kelvin": None,
            "minimum_celsius": None, "minimum_kelvin": None,
            "maximum_celsius": None, "maximum_kelvin": None,
            "area": 0
        }

    # Get the center and radius from the selected_coords
    (x_center, y_center), radius = coords

    # Create a grid of points to check against the circle equation
    Y, X = np.indices(data.shape)
    distance_from_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

    # Create a mask for points within the circle
    mask = distance_from_center <= radius
    region_data = data[mask]

    # Remove NaN values
    region_data = region_data[~np.isnan(region_data)]

    if region_data.size == 0:
        return {
            "average_celsius": None, "average_kelvin": None,
            "minimum_celsius": None, "minimum_kelvin": None,
            "maximum_celsius": None, "maximum_kelvin": None,
            "area": 0
        }

    # Calculate statistics
    avg_temp_celsius = np.mean(region_data)
    min_temp_celsius = np.min(region_data)
    max_temp_celsius = np.max(region_data)

    # Highlight the circular region on the image
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='inferno', interpolation='nearest')
    circle_patch = plt.Circle((x_center, y_center), radius, fill=False, linewidth=2)
    ax.add_patch(circle_patch)

    # Display analysis results
    result_text = (
        f"Area: {region_data.size} pixels\n"
        f"Avg Temp: {avg_temp_celsius:.2f} °C / {avg_temp_celsius + 273.15:.2f} K\n"
        f"Min Temp: {min_temp_celsius:.2f} °C / {min_temp_celsius + 273.15:.2f} K\n"
        f"Max Temp: {max_temp_celsius:.2f} °C / {max_temp_celsius + 273.15:.2f} K"
    )
    ax.text(
        0.05, 0.95, result_text, transform=ax.transAxes,
        fontsize=10, color='white', verticalalignment='top',
        bbox=dict(alpha=0.5)
    )
    plt.title("Circle Analysis")
    plt.show()

    return {
        "average_celsius": avg_temp_celsius,
        "average_kelvin": avg_temp_celsius + 273.15,
        "minimum_celsius": min_temp_celsius,
        "minimum_kelvin": min_temp_celsius + 273.15,
        "maximum_celsius": max_temp_celsius,
        "maximum_kelvin": max_temp_celsius + 273.15,
        "area": region_data.size
    }

# Global variables reset to avoid residual states
selected_coords = []
toggle = None

def reset_globals():
    """
    Reset global variables to their default state.
    """
    global selected_coords, toggle
    selected_coords = []
    toggle = None

def interactive_draw(data, shape):
    """
    Allows the user to draw a shape interactively on the plot and displays analysis results.

    Args:
        data (numpy.ndarray): The temperature data.
        shape (str): The type of shape ('line', 'rectangle',  'circle').

    Returns:
        list: The coordinates for the selected shape.
    """
    global selected_coords

    # Reset global variables
    reset_globals()

    # Close any existing figures to avoid unnecessary popups
    plt.close('all')

    # Create one figure for interactive drawing
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap='inferno', interpolation='nearest')

    # Add colorbar for temperature
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', location='left')
    cbar.set_label('Temperature (°C)', fontsize=10)

    ax.set_title(f"Draw a {shape.capitalize()}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Initialize appropriate shape selector
    selector = None
    if shape == "line":
        selector = RectangleSelector(ax, line_selector_callback, useblit=True, button=[1], interactive=True)
    elif shape == "rectangle":
        selector = RectangleSelector(ax, rectangle_selector_callback, useblit=True, button=[1], interactive=True)
    elif shape == "circle":
        selector = EllipseSelector(ax, ellipse_selector_callback, useblit=True, button=[1], interactive=True)
    else:
        raise ValueError("Invalid shape type.")

    # Display the interactive plot
    plt.show()

    # Deactivate the selector
    if selector:
        selector.set_active(False)

    # Perform analysis only if coordinates are selected
    if selected_coords:
        # Close the previous plot
        plt.close('all')

        # Analyze the selected region
        analysis = analyze_region(data, shape, selected_coords)

        # Create the final plot for results
        fig, ax = plt.subplots()
        img = ax.imshow(data, cmap='inferno', interpolation='nearest')
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', location='left')
        cbar.set_label('Temperature (°C)', fontsize=10)

        # Highlight the selected shape
        if shape == "line":
            (x1, y1), (x2, y2) = selected_coords
            ax.plot([x1, x2], [y1, y2], color='white', linestyle='-', linewidth=2)
        elif shape == "rectangle":
            (x1, y1), (x2, y2) = selected_coords
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='white', linewidth=2))
        elif shape == "circle":
            (x_center, y_center), radius = selected_coords
            circle = plt.Circle((x_center, y_center), radius, fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(circle)

        # Display analysis results
        if analysis["average_celsius"] is not None:
            result_text = (
                f"Area: {analysis['area']} pixels\n"
                f"Average Temp: {analysis['average_celsius']:.2f} °C / {analysis['average_kelvin']:.2f} K\n"
                f"Min Temp: {analysis['minimum_celsius']:.2f} °C / {analysis['minimum_kelvin']:.2f} K\n"
                f"Max Temp: {analysis['maximum_celsius']:.2f} °C / {analysis['maximum_kelvin']:.2f} K"
            )
        else:
            result_text = "No valid data in the selected region."

        # Add result text to the plot
        ax.text(
            1.02, 0.5, result_text, transform=ax.transAxes,
            fontsize=10, color='black', verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8),
            clip_on=False
        )

        plt.title(f"{shape.capitalize()} Analysis")
        plt.show()

    return selected_coords

def line_selector_callback(eclick, erelease):
    """
    Callback for selecting a line.
    """
    global selected_coords
    selected_coords = [(int(eclick.xdata), int(eclick.ydata)), (int(erelease.xdata), int(erelease.ydata))]
    plt.close()

def rectangle_selector_callback(eclick, erelease):
    """
    Callback for selecting a rectangle.
    """
    global selected_coords
    selected_coords = [(int(eclick.xdata), int(eclick.ydata)), (int(erelease.xdata), int(erelease.ydata))]
    plt.close()

def ellipse_selector_callback(eclick, erelease):
    """
    Callback for selecting a circle.
    """
    global selected_coords, toggle

    if toggle is None:  # First click: store the center of the circle
        toggle = (int(eclick.xdata), int(eclick.ydata))
    else:  # Second click: calculate the radius
        x_center, y_center = toggle
        radius = int(np.sqrt((eclick.xdata - x_center) ** 2 + (eclick.ydata - y_center) ** 2))
        selected_coords = [(x_center, y_center), radius]
        toggle = None  # Reset toggle for the next interaction
        plt.close()  # Close the plot after selection


def display_image_on_canvas_with_analysis(data, canvas, title, analysis=None):
    """
    Display the image on a Tkinter canvas, fully fitting the canvas size, with analysis and taskbar.

    Args:
        data (numpy.ndarray): Data to display.
        canvas (tk.Canvas): Tkinter canvas to use.
        title (str): Title of the image.
        analysis (dict): Analysis results to display (optional).
    """
    if data is None or data.size == 0:
        canvas.create_text(200, 200, text="No valid data", fill="red", font=("Arial", 16))
        return

    # Remove any previous figure from the canvas
    for child in canvas.winfo_children():
        child.destroy()

    # Dynamically determine figure size based on data dimensions
    aspect_ratio = data.shape[1] / data.shape[0]  # Columns / Rows
    fig_width = 6  # Standard figure width in inches
    fig_height = fig_width / aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data, cmap='inferno', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    fig.colorbar(im, ax=ax, orientation='vertical', label='Intensity')

    # Embed the matplotlib figure into the Tkinter canvas
    canvas_figure = FigureCanvasTkAgg(fig, master=canvas)
    canvas_figure.draw()

    # Add the widget to the canvas
    canvas_widget = canvas_figure.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Add shape analysis results if available
    if analysis:
        shape_analysis_text = "\nShape Analysis Results:\n"
        
        # Get parameters from analysis dictionary
        for key, value in analysis.items():
            shape_analysis_text += f"{key}: {value}\n"

        # Display the analysis below the image
        analysis_label = tk.Label(canvas, text=shape_analysis_text, font=("Arial", 10), justify="left")
        analysis_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    # Add a taskbar
    toolbar_frame = tk.Frame(canvas)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas_figure, toolbar_frame)
    toolbar.update()


def analyze_shape(data):
    """
    Perform shape analysis on the given image data.
    Extracts properties like bounding box, area, and min/max coordinates.

    Args:
        data (numpy.ndarray): Image data to analyze.

    Returns:
        dict: Analysis results containing shape parameters.
    """
    # Label connected components in the image (binary or segmented)
    labeled_image = measure.label(data)
    
    # Get region properties
    regions = measure.regionprops(labeled_image)
    
    # If no regions are detected, return an empty dictionary
    if not regions:
        return {
            'Bounding Box': 'N/A',
            'Area': 'N/A',
            'Min Intensity': 'N/A',
            'Max Intensity': 'N/A'
        }

    # Calculate the analysis results for the largest region (or any specific region you prefer)
    region = regions[0]
    bounding_box = region.bbox  # (min_row, min_col, max_row, max_col)
    area = region.area
    min_intensity = np.min(data[region.slice])
    max_intensity = np.max(data[region.slice])

    return {
        'Bounding Box': f"{bounding_box}",
        'Area': area,
        'Min Intensity': min_intensity,
        'Max Intensity': max_intensity
    }


def modify_image_with_constant(data, result_data, canvas_to_update, title):
    """
    Opens a new window to allow the user to input a constant value and perform an arithmetic operation,
    then updates the image with shape analysis results.

    Args:
        data (numpy.ndarray): Dataset to modify.
        result_data (numpy.ndarray): Result dataset to reflect the changes.
        canvas_to_update (tk.Canvas): Canvas to update with the modified image.
        title (str): Title of the image ("Image 1" or "Image 2").
    """
    input_window = tk.Toplevel()
    input_window.title(f"Modify {title}")
    input_window.geometry("500x600")

    # Title Label
    tk.Label(input_window, text=f"Enter a constant value and select an operation for {title}", font=("Arial", 12)).pack(pady=10)

    # Input Field for the Constant Value
    constant_value_entry = tk.Entry(input_window, font=("Arial", 12))
    constant_value_entry.pack(pady=10)

    # Dropdown Menu for Selecting Operation
    operation_var = tk.StringVar()
    operation_dropdown = ttk.Combobox(input_window, textvariable=operation_var)
    operation_dropdown['values'] = ["Add", "Subtract", "Multiply", "Divide"]
    operation_dropdown.set("Select Operation")
    operation_dropdown.pack(pady=10)

    # Result Canvas for displaying the modified image
    result_canvas = tk.Canvas(input_window, width=400, height=400)
    result_canvas.pack(pady=20)

    def apply_operation():
        try:
            constant_value = float(constant_value_entry.get())  # Get constant value from entry
            operation = operation_var.get()  # Get selected operation

            if operation == "Add":
                result = np.add(data, constant_value)
            elif operation == "Subtract":
                result = np.subtract(data, constant_value)
            elif operation == "Multiply":
                result = np.multiply(data, constant_value)
            elif operation == "Divide":
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.divide(data, constant_value)
                    result[np.isnan(result)] = 0  # Replace NaNs with 0
            else:
                messagebox.showwarning("Warning", "Please select a valid operation!")
                return

            # Update result data
            result_data[:, :] = result

            # Update canvas with the modified image and analysis text
            display_image_on_canvas_with_analysis(result, result_canvas, f"{title} Modified")
            
            print("Choose the shape for analysis: line, rectangle, circle")
            shape = input("Enter the shape (line/rectangle/circle): ").lower()
            if shape in ["line", "rectangle", "polygon", "circle"]:
                coords = interactive_draw(result_data, shape)
                analysis = analyze_region(result_data, shape, coords)
                print(f"\n{shape.capitalize()} Analysis:")
                if analysis["average_celsius"] is not None:
                    print(f"  Area: {analysis['area']} pixels")
                    print(f"  Average Temperature: {analysis['average_celsius']:.2f} °C / {analysis['average_kelvin']:.2f} K")
                    print(f"  Minimum Temperature: {analysis['minimum_celsius']:.2f} °C / {analysis['minimum_kelvin']:.2f} K")
                    print(f"  Maximum Temperature: {analysis['maximum_celsius']:.2f} °C / {analysis['maximum_kelvin']:.2f} K")
                else:
                    print("  No valid data in the selected region.")
            else:
                print("Invalid shape selected.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply operation: {e}")

    # Apply Button
    tk.Button(input_window, text="Apply Operation", command=apply_operation).pack(pady=10)

    # Close Button
    tk.Button(input_window, text="Close", command=input_window.destroy).pack(pady=10)


def process_files():
    """
    Main function to process files, display images, and handle operations with integrated analysis.
    """
    # File selection
    file1 = filedialog.askopenfilename(title="Select First Excel File", filetypes=[("Excel files", "*.xlsx")])
    file2 = filedialog.askopenfilename(title="Select Second Excel File", filetypes=[("Excel files", "*.xlsx")])

    if not file1 or not file2:
        messagebox.showwarning("Warning", "Both files must be selected!")
        return

    # Preprocess data
    data1 = preprocess_data(file1)
    data2 = preprocess_data(file2)

    if data1 is None or data2 is None:
        return

    # Trim data to common shape
    data1, data2 = trim_to_common_shape(data1, data2)

    # Create a new window to display the images
    operation_window = tk.Toplevel()
    operation_window.title("Image Analysis and Operations")
    operation_window.geometry("1000x600")


    canvas1 = tk.Canvas(operation_window, width=400, height=400)
    canvas1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    display_image_on_canvas_with_analysis(data1, canvas1, "Image 1")

    canvas2 = tk.Canvas(operation_window, width=400, height=400)
    canvas2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    display_image_on_canvas_with_analysis(data2, canvas2, "Image 2")

    # Dropdown for selecting operation
    operation_var = tk.StringVar()
    operation_dropdown = ttk.Combobox(operation_window, textvariable=operation_var)
    operation_dropdown['values'] = ["Add", "Subtract", "Multiply", "Divide"]
    operation_dropdown.set("Select Operation")
    operation_dropdown.pack(pady=10)

    # Function to handle result popup
    def show_result_popup():
        operation = operation_var.get()
        if operation:
            result = perform_operation(data1, data2, operation)
            if result is not None:
                result_analysis = analyze_region(result, "rectangle", [(0, 0), (result.shape[1] - 1, result.shape[0] - 1)])
                result_window = tk.Toplevel()
                result_window.title(f"Result of {operation}")
                result_window.geometry("600x600")
                result_canvas = tk.Canvas(result_window, width=600, height=600)
                result_canvas.pack(fill=tk.BOTH, expand=True)
                display_image_on_canvas_with_analysis(result, result_canvas, f"Result of {operation}")

                print("Choose the shape for analysis: line, rectangle, or circle")
                shape = input("Enter the shape (line/rectangle/circle): ").lower()

                if shape in ["line", "rectangle",  "circle"]:
                    coords = interactive_draw(result, shape)
                    analysis = analyze_region(result, shape, coords)

                    print(f"\n{shape.capitalize()} Analysis:")
                    if analysis["average_celsius"] is not None:
                        print(f"  Area: {analysis['area']} pixels")
                        print(f"  Average Temperature: {analysis['average_celsius']:.2f} °C / {analysis['average_kelvin']:.2f} K")
                        print(f"  Minimum Temperature: {analysis['minimum_celsius']:.2f} °C / {analysis['minimum_kelvin']:.2f} K")
                        print(f"  Maximum Temperature: {analysis['maximum_celsius']:.2f} °C / {analysis['maximum_kelvin']:.2f} K")
                    else:
                        print("  No valid data in the selected region.")
                else:
                    print("Invalid shape selected.")
    
                
    # Show Result Button
    tk.Button(operation_window, text="Show Result", command=show_result_popup).pack(pady=10)

    button_frame = tk.Frame(operation_window)
    button_frame.pack(pady=10)

    # Modify Image 1 Button
    tk.Button(button_frame, text="Modify Image 1", command=lambda: modify_image_with_constant(data1, data1, canvas1, "Image 1")).grid(row=0, column=0, padx=10)
    # Modify Image 2 Button
    tk.Button(button_frame, text="Modify Image 2", command=lambda: modify_image_with_constant(data2, data2, canvas2, "Image 2")).grid(row=0, column=1, padx=10)

# Main application window
root = tk.Tk()
root.title("Arithmetic Operations on IR Data")
root.geometry("400x200")

tk.Label(root, text="Perform Arithmetic Operations on Excel Data", font=("Arial", 14)).pack(pady=20)
tk.Button(root, text="Select Files and Perform Operations", command=process_files).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

root.mainloop()