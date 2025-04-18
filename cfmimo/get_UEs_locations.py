import sys
import io
import json
import re
import numpy as np
from PyQt5 import QtWidgets, QtWebEngineWidgets
import folium
from folium.plugins import Draw

class WebEnginePage(QtWebEngineWidgets.QWebEnginePage):
    """ Custom QWebEnginePage to handle JavaScript console messages. """

    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        """ Capture JavaScript messages and process the coordinates. """
        try:
            coords_dict = json.loads(msg)  # Convert JSON string to dictionary
            coords = coords_dict['geometry']['coordinates'][0]  # Extract coordinates
            print("Extracted Coordinates:", coords)
            np.save('UEcoords.npy', coords)  # Save coordinates as .npy file
        except json.JSONDecodeError:
            print("Error decoding JSON message.")

def main():
    """ Main function to create and launch the PyQt5 application. """
    app = QtWidgets.QApplication(sys.argv)

    # Initialize Folium map
    map_center = [45.19026, 5.71946]  # Grenoble, France
    m = folium.Map(location=map_center, zoom_start=17)

    # Add drawing tools to the map
    draw = Draw(
        draw_options={
            'polyline': True,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': True,
            'circlemarker': False
        },
        edit_options={'edit': True}
    )
    m.add_child(draw)

    # Save the map to an HTML data buffer
    data = io.BytesIO()
    m.save(data, close_file=False)

    # Convert HTML content to string
    html_content = data.getvalue().decode()

    # Extract the correct map variable name from Folium-generated HTML
    match = re.search(r"var\s+(map_\w+)\s*=\s*L\.map", html_content)
    if match:
        map_variable = match.group(1)  # Get the actual map variable name
    else:
        print("Error: Could not find the Folium map variable name.")
        return

    # Inject JavaScript to capture draw events and log coordinates
    injected_script = f"""
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                {map_variable}.on("draw:created", function(event) {{
                    console.log(JSON.stringify(event.layer.toGeoJSON())); 
                }});
            }});
        </script>
    """

    # Inject JavaScript into HTML
    modified_html = html_content.replace("</body>", injected_script + "</body>")

    # Create a PyQt5 Web View to display the map
    view = QtWebEngineWidgets.QWebEngineView()
    page = WebEnginePage(view)
    view.setPage(page)
    view.setHtml(modified_html)  # Load the modified HTML
    view.show()

    sys.exit(app.exec_())  # Run the PyQt5 application

if __name__ == '__main__':
    main()
