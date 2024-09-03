from myspec.model import get_bioimage_model_v4
from bioimageio.spec import save_bioimageio_package, load_description
from bioimageio.core import create_prediction_pipeline



model = get_bioimage_model_v4()

zip_file = save_bioimageio_package(model)
ret_model = load_description(zip_file)

prediction_pipeline = create_prediction_pipeline(bioimageio_model=ret_model)