import os
from flask import Flask
from flask import render_template
import fasttext
from week4.foo_classification import get_category_id_name_map
from pathlib import Path


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
        QUERY_CLASS_MODEL_LOC = os.environ.get("QUERY_CLASS_MODEL_LOC", "/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/queries/fasttext_1000-epoch-50_lr-0.1_dim-100_minn-0_maxn-3_wordNgrams-2")
        if QUERY_CLASS_MODEL_LOC and os.path.isfile(QUERY_CLASS_MODEL_LOC):
            app.config["query_model"] = fasttext.load_model(QUERY_CLASS_MODEL_LOC)
        else:
            print("No query model found.  Have you run fasttext?")
        print("QUERY_CLASS_MODEL_LOC: %s" % QUERY_CLASS_MODEL_LOC)
        CATEGORY_HIERARCHY_FILE = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml")
        category_id_name_map = get_category_id_name_map(CATEGORY_HIERARCHY_FILE)
        app.config["category_id_name_map"] = category_id_name_map
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    app.config["index_name"] = os.environ.get("INDEX_NAME", "bbuy_products")

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    # A simple landing page
    # @app.route('/')
    # def index():
    #    return render_template('index.jinja2')
    from . import search
    app.register_blueprint(search.bp)
    app.add_url_rule('/', view_func=search.query)
    return app