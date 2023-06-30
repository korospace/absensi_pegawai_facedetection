from flask_restful import Api
from face_detection.views import CheckEmployeeFolder, CreateModelFile, CompareFace

def face_detection_routes(api: Api):
    api.add_resource(CheckEmployeeFolder, "/api/check_employee_folder")

    api.add_resource(CreateModelFile, "/api/create_model_file")

    api.add_resource(CompareFace, "/api/compare_face")
