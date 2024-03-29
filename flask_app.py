from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
import os

path = 'C:\\Workspace\\smart_surveillance\\centroid_simple-object-tracking\\frame_captures'

app = Flask(__name__)

# THIS IS THE SERVER
# TODO: instead of displaying the image, just be able to save the file somewhere
@app.route('/')
def index():
    return 'Hello, World!'
    # response = send_from_directory(directory='your-directory', filename='your-file-name')
    # response.headers['my-custom-header'] = 'my-custom-status-0'
    # return response

def get_captures():
    return send_file('frame_captures/object0.jpg')
    # response = send_from_directory(directory='your-directory', filename='your-file-name')
    # response.headers['my-custom-header'] = 'my-custom-status-0'
    # return response

@app.route('/send_files/')
def send_files():
    objects = [0,1]
    # objects = []
    # # r=root, d=directories, f = objects
    # for r, d, f in os.walk(path):
    #     for file in f:
    #         objects.append(os.path.join(r, file))

    # for f in objects:
    #     print(f)
    return render_template('files.html', objects=objects)

@app.route('/submit/', methods=['POST'])
def submit():
    content = request.json
    print(content['object_id'])
    print("frame received")
    # print(content['image'])
    #TODO: work on what is returned here
    # return None
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)