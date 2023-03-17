# coding=utf-8
# Copyright 2023 The R-U-SURE Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for rendering DAGs in Colab/IPython notebooks."""

from typing import Optional, Iterable

from IPython import display

from r_u_sure.decision_diagrams import gated_state_dag
from r_u_sure.rendering import dag_annotator
from r_u_sure.rendering import svg_renderer



def enable_notebook_interactive_features(
    max_zoom: int = 100,
    zoom: bool = True,
    hover_raise: bool = True,
    max_width: Optional[str] = "50em",
    max_height: Optional[str] = "100em",
) -> None:
  """Constructs a script to enable panning and zooming in IPython."""
  if max_width:
    script = f"""
    const svgElts = document.querySelectorAll(".rendered_graph");
    const svgElt = svgElts[svgElts.length - 1];
    svgElt.style.maxWidth = "{max_width}";
    """
    display.display(display.Javascript(data=script))

  if max_height:
    script = f"""
    const svgElts = document.querySelectorAll(".rendered_graph");
    const svgElt = svgElts[svgElts.length - 1];
    svgElt.style.maxHeight = "{max_height}";
    """
    display.display(display.Javascript(data=script))

  if zoom:
    url = "https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/src/svg-pan-zoom.js"  # pylint:disable=line-too-long
    display.display(display.Javascript(url=url))

    zoom_script = (
        f"""
        const maxZoom = {max_zoom};
        """
        + """
        const svgElts = document.querySelectorAll(".rendered_graph");
        const svgElt = svgElts[svgElts.length - 1];
        svgElt.style.width = "100%";
        svgElt.style.height = "50vw";
        const panZoom = svgPanZoom(svgElt, {
            controlIconsEnabled: true,
            maxZoom: maxZoom,
        });
        """
    )
    display.display(display.Javascript(data=zoom_script))

  if hover_raise:
    hover_script = """
        const hoverRaiseGroups = document.getElementsByClassName("hover_to_top");
        function handleHoverRaise(element) {
            // Move the child to the end of its parent, raising it to the top of
            // the draw order.
            const parent = element.parentElement;
            element.remove();
            parent.appendChild(element);
        }
        for (const elt of hoverRaiseGroups) {
            elt.onmouseover = () => handleHoverRaise(elt);
        }
        """
    display.display(display.Javascript(data=hover_script))


def render_dag_in_notebook(
    dag: gated_state_dag.CompleteStateDAG,
    annotator: dag_annotator.StateDAGAnnotator,
    hover_for_info: bool = True,
    pan_and_zoom: bool = False,
    emphasized_edges: Optional[Iterable[gated_state_dag.Edge]] = None,
    max_width: Optional[str] = "50em",
    max_height: Optional[str] = "100em",
):
  """Renders a DAG in a Colab/IPython notebook.

  Args:
    dag: The DAG to render.
    annotator: An annotator for this DAG.
    hover_for_info: Whether to enable hovering for extra info. Greatly increases
      the size of the output HTML, so only recommended for small graphs.
    pan_and_zoom: Whether to enable pan and zoom functionality.
    emphasized_edges: An optional list of edges to emphasize during drawing, for
      instance, to visualize the shortest path through the graph.
    max_width: Maximum width for the rendered cell.
    max_height: Maximum height for the rendered cell.
  """
  renderer = svg_renderer.SVGRenderer(
      with_default_styles=True, with_hover_text=hover_for_info
  )
  dag_annotator.render_dag(
      dag=dag,
      annotator=annotator,
      renderer=renderer,
      emphasized_edges=emphasized_edges,
  )

  display.display(display.HTML(data=renderer.to_html(padding=10)))
  enable_notebook_interactive_features(
      zoom=pan_and_zoom,
      hover_raise=True,
      max_width=max_width,
      max_height=max_height,
  )
