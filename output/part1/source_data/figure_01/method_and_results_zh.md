# HUP060 Part 1: connected PLV network and actuator selection

- Graph: training-only equal-weight multiband PLV; maximum-spanning-tree backbone plus the strongest edges to density 0.10.
- Size: 36 nodes, 63 edges, 1 connected component.
- Score: `0.30 degree + 0.60 betweenness + 0.10 eigenvector`; each component is min-max normalized within HUP060.
- Threshold: training-score 65th percentile, raw cutoff `0.288218`; selection uses a strict greater-than rule. The actuator-density hyperparameter was selected during development and is fixed before frozen-test evaluation.
- Clinical labels: SOZ/resection labels are excluded from graph construction, scoring, and threshold selection; they are used only for retrospective coloring.
- Red nodes (selected and SOZ/resection matched, n=5): RPFa1, RPFa2, RPFa3, RPFb1, RPFc1.
- Blue nodes (selected without SOZ/resection match, n=8): LAF2, RA3, RAFa1, RAFb1, RAFb2, RAFc2, RAFc3, RAFd1.
- All unselected nodes are grey.

Descriptive overlap: 5/13 selected nodes match the SOZ/resection union, covering 5/10 clinically labelled nodes. This is a one-patient retrospective description, not an inferential localization result.
