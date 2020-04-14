"""Thanks to https://gist.github.com/Zsailer/818b971cd469f6a055a2844199581795
"""
import collections

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def main();
   if sdsd:
         print("sdsds")


#  if nodelist is None:
#      nodelist = list(G)

#  if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
#      return None

#  try:
#      xy = np.asarray([pos[v] for v in nodelist])
#  except KeyError as e:
#      raise nx.NetworkXError('Node %s has no position.' % e)
#  except ValueError:
#      raise nx.NetworkXError('Bad value in node positions.')

#  if isinstance(alpha, collections.Iterable):
#      node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
#      alpha = None

#  if cmap is not None:
#      cm = mpl.cm.get_cmap(cmap)
#      norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#  else:
#      cm = None
#      norm = None
#  print(node_width)

#  node_collection = mpl.collections.EllipseCollection(
#      widths=node_width,
#      heights=node_height,
#      angles=0,
#      offsets=np.array(xy),
#      cmap=cm,
#      norm=norm,
#      transOffset=ax.transData,
#      linewidths=linewidths)

#  node_collection.set_array(node_color)
#  node_collection.set_label(label)
#  node_collection.set_zorder(2)
#  ax.add_collection(node_collection)
#  plt.show()

#  return node_collection
