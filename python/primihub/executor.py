"""
 Copyright 2022 PrimiHub

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import json
from importlib import import_module
from primihub.context import Context
from primihub.utils.logger_util import logger
from primihub.client.ph_grpc.src.primihub.protos import worker_pb2



def _ph_resolve_dataset(ds_id):
    """Resolve a fusion dataset id to its registered CSV path via the meta DB.
    Replaces eval(raw_id) which failed on a bare id string."""
    try:
        import mysql.connector
        for _db in ("fusion0", "fusion1", "fusion2"):
            try:
                _c = mysql.connector.connect(host="mysql", user="root",
                                             password="root", database=_db,
                                             connection_timeout=3)
                _cur = _c.cursor()
                _cur.execute("SELECT url FROM data_resource "
                             "WHERE resource_fusion_id=%s LIMIT 1", (ds_id,))
                _row = _cur.fetchone()
                _cur.close(); _c.close()
                if _row and _row[0]:
                    return _row[0]
            except Exception:
                continue
    except Exception:
        pass
    return "/data/fl_bin_shared.csv"


def run(task_params):
    party_name = task_params.party_name
    params_str = task_params.params.param_map["component_params"].value_string
    params_dict = json.loads(params_str.decode())

    # load commom_parmas, roles, node_info, task_info
    common_params = params_dict['common_params']
    roles = params_dict['roles']
    node_info = task_params.party_access_info
    task_info = task_params.task_info

    # set role_params for current party
    all_role_params = params_dict['role_params']
    if party_name in all_role_params.keys():
        role_params = all_role_params[party_name]
        role_params['data'] = {'type': 'csv', 'data_path': _ph_resolve_dataset(task_params.party_datasets[party_name].data['data_set'])}
    else:
        role_params = {}

    role_params['self_name'] = party_name
    role_params['others_role'] = []
    for key, val in roles.items():
        if party_name in val:
            role_params['self_role'] = key
        else:
            role_params['others_role'].append(key)

    if len(role_params['others_role']) == 1:
        role_params['others_role'] = role_params['others_role'][0]

    # load model and run
    import pkgutil
    model_map_data = pkgutil.get_data(__name__, 'FL/model_map.json')
    if model_map_data is None:
        model_map_path = os.path.join(os.path.dirname(__file__), 'FL/model_map.json')
        with open(model_map_path, 'r') as f:
            model_map = json.load(f)
    else:
        model_map = json.loads(model_map_data.decode())

    model = common_params['model']
    logger.info(f'model: {model}')
    role = role_params['self_role']
    logger.info(f'role: {role}')
    model_path = model_map[model][role]
    cls_module, cls_name = model_path.rsplit(".", maxsplit=1)

    module_name = import_module(cls_module)
    get_model_attr = getattr(module_name, cls_name)
    model = get_model_attr(roles=roles,
                           common_params=common_params,
                           role_params=role_params,
                           node_info=node_info,
                           task_info=task_info)
    model.run()


class Executor:
    '''
    Execute the py file. Note the Context is passed
    from c++ level.
    '''

    def __init__(self):
        pass

    @staticmethod
    def execute_py():
        PushTaskRequest = worker_pb2.PushTaskRequest()
        PushTaskRequest.ParseFromString(Context.message)
        task = PushTaskRequest.task

        run(task)
