# core import
from flask import Response
from flask_restful import Resource
from flask import request, make_response

# service import
from face_detection.service import check_employee_folder, create_model_file, compare_faces

class CheckEmployeeFolder(Resource):
    @staticmethod
    def get() -> Response:
        response, status = check_employee_folder(request)

        return make_response(response, status)

class CreateModelFile(Resource):
    @staticmethod
    def post() -> Response:
        response, status = create_model_file(request)

        return make_response(response, status)

class CompareFace(Resource):
    @staticmethod
    def post() -> Response:
        response, status = compare_faces(request)

        return make_response(response, status)