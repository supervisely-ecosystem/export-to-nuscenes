from pathlib import Path

import supervisely as sly

import src.functions as f
import src.globals as g


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        project = g.api.project.get_info_by_id(id=context.project_id)
        nuscenes_path = Path(g.app_data).joinpath(f"{project.id}_{project.name}_nuscenes")
        nuscenes_path.mkdir(exist_ok=True)

        return f.convert_sly_project_to_nuscenes(g.api, context.project_id, nuscenes_path)


def main():
    app = MyExport()
    app.run()
    if not sly.is_development():
        sly.fs.remove_dir(g.app_data)


if __name__ == "__main__":
    sly.main_wrapper("main", main, log_for_agent=False)
