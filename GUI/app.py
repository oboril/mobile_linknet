from flask import Flask, request, render_template, url_for, abort, redirect
import json
import os

from plotly.offline import plot

print("Loading model")
import mobile_linknet as ml
import numpy as np
SIZE = (224*4,224*3)

model = ml.Mobile_LinkNet_SAM(load_saved=os.getcwd()+"/model.h5")

print("Model loaded")

app = Flask(__name__, static_url_path='', template_folder=os.getcwd(), static_folder=os.getcwd())

app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.get("/")
@app.get("/index")
def index():
    return render_template("index.html")

@app.get("/images/raw<id>")
def raw_image(id):
    if uploaded_image is None:
        abort(400)
    return redirect(url_for('static',filename=uploaded_image))

@app.get("/images/masks<id>")
def image_masks(id):
    return redirect(url_for('static',filename="temp/masks.png"))

@app.get("/images/centers<id>")
def image_centers(id):
    return redirect(url_for('static',filename="/temp/centers.png"))

@app.post("/process-image")
def process_image():
    if uploaded_image is None:
        abort(400)
    
    print("Image width", image_width)

    image = ml.utils.open_image(uploaded_image)
    predicted = ml.postprocessing.get_prediction(model, image)

    cells, masks = ml.postprocessing.segment_cells(predicted, cells_smoothing_sigma=0.01)

    overlay = ml.postprocessing.transparent_segmentation_masks(masks)

    confluency = np.sum(overlay[:,:,3])/overlay.shape[0]/overlay.shape[1]

    ml.postprocessing.save_cell_centers(cells,SIZE,os.getcwd()+"/temp/centers.png")
    ml.utils.save_image(os.getcwd()+"/temp/masks.png",overlay)

    pix2um = float(image_width)/SIZE[0]
    global regionprops
    regionprops = ml.postprocessing.get_statistics(masks, pix2um)
    
    regionprops_str = {}
    for key1 in regionprops:
        regionprops_str[key1] = {}
        for key2 in regionprops[key1]:
            if key2 != "raw_data":
                regionprops_str[key1][key2] = f"{regionprops[key1][key2]:0.0f}"

    return json.dumps({'success':True, "cells":len(cells), "confluency":f"{confluency*100:0.1f} %" , "regionprops": regionprops_str})

uploaded_image = None
image_width = None
regionprops = None
@app.post("/upload-image")
def upload_image():
    if 'file' not in request.files:
        print("No file selected")
        return json.dumps({'success':False})
    
    f = request.files['file']

    print("Uploading image", f.filename)

    global image_width
    image_width = request.values["width"]

    global uploaded_image
    uploaded_image = "temp/uploaded_image." + f.filename.split(".")[-1]
    f.save(os.getcwd()+"/"+uploaded_image)

    image = ml.utils.open_image(uploaded_image, SIZE)
    uploaded_image = "temp/uploaded_image.jpg"
    ml.utils.save_image(os.getcwd()+"/"+uploaded_image, image)

    return json.dumps({'success':True})

@app.post("/generate-plot")
def generate_plot():
    global regionprops
    if regionprops is None:
        abort(400)

    names = {"area":"Area","eccentricity":"Eccentricity","major_axis":"Long axis","minor_axis":"Short axis","orientation":"Orientation"}
    data = json.loads(request.data)

    plot = generate_plotly(regionprops[data["xaxis"]]["raw_data"], regionprops[data["yaxis"]]["raw_data"], names[data["xaxis"]], names[data["yaxis"]])

    return json.dumps({'success':True, "plot":plot})

def generate_plotly(dataX, dataY, labelX, labelY):
    fig = {
        'data': [{'legendgroup': '',
                  'marker': {'color': '#636efa', 'symbol': 'circle'},
                  'mode': 'markers',
                  'name': '',
                  'orientation': 'v',
                  'showlegend': False,
                  'type': 'scatter',
                  'x': dataX,
                  'xaxis': 'x',
                  'y': dataY,
                  'yaxis': 'y',
                  'hoverinfo': 'skip'},
                 {'alignmentgroup': 'True',
                  'legendgroup': '',
                  'marker': {'color': '#636efa', 'symbol': 'circle'},
                  'name': '',
                  'offsetgroup': '',
                  'scalegroup': 'x',
                  'showlegend': False,
                  'type': 'violin',
                  'x': dataX,
                  'xaxis': 'x3',
                  'yaxis': 'y3',
                  'hoverinfo': 'skip'},
                 {'alignmentgroup': 'True',
                  'legendgroup': '',
                  'marker': {'color': '#636efa', 'symbol': 'circle'},
                  'name': '',
                  'offsetgroup': '',
                  'showlegend': False,
                  'type': 'violin',
                  'xaxis': 'x2',
                  'y': dataY,
                  'yaxis': 'y2',
                  'hoverinfo': 'skip'}],
            'layout': {'legend': {'tracegroupgap': 0},
                   "margin":dict(l=0, r=0, t=0, b=0),
                   'template': 'plotly_white',
                   'xaxis': {'anchor': 'y', 'domain': [0.0, 0.8358], 'title': {'text': labelX}},
                   'xaxis2': {'anchor': 'y2',
                              'domain': [0.8408, 1.0],
                              'matches': 'x2',
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': False,
                              'ticks': ''},
                   'xaxis3': {'anchor': 'y3',
                              'domain': [0.0, 0.8358],
                              'matches': 'x',
                              'showgrid': True,
                              'showticklabels': False},
                   'xaxis4': {'anchor': 'y4',
                              'domain': [0.8408, 1.0],
                              'matches': 'x2',
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': False,
                              'ticks': ''},
                   'yaxis': {'anchor': 'x', 'domain': [0.0, 0.8316], 'title': {'text': labelY}},
                   'yaxis2': {'anchor': 'x2',
                              'domain': [0.0, 0.8316],
                              'matches': 'y',
                              'showgrid': True,
                              'showticklabels': False},
                   'yaxis3': {'anchor': 'x3',
                              'domain': [0.8416, 1.0],
                              'matches': 'y3',
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': False,
                              'ticks': ''},
                   'yaxis4': {'anchor': 'x4',
                              'domain': [0.8416, 1.0],
                              'matches': 'y3',
                              'showgrid': True,
                              'showline': False,
                              'showticklabels': False,
                              'ticks': ''}}}
    
    return plot(fig, include_plotlyjs=True, output_type='div')

app.run()