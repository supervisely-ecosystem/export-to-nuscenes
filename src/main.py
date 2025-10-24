from pathlib import Path

import supervisely as sly

import src.functions as f
import src.globals as g


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        project = g.api.project.get_info_by_id(id=context.project_id)
        nuscenes_path = Path(g.app_data).joinpath(f"{project.id}_{project.name}_nuscenes")
        nuscenes_path.mkdir(exist_ok=True)
        f.convert_sly_project_to_nuscenes(g.api, context.project_id, nuscenes_path)
        return nuscenes_path.as_posix()


def main():
    try:
        app = MyExport()
        app.run()
    except Exception as e:
        f.handle_exception(e, g.api, g.task_id)
    finally:
        if not sly.is_development():
            sly.fs.remove_dir(g.app_data)


if __name__ == "__main__":
    sly.main_wrapper("main", main, log_for_agent=False)
