"""Differentiable atom-centered symmetry functions used by TeNNPot."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from ase.neighborlist import primitive_neighbor_list
from torch import nn


def _cell_tensor(cell, *, dtype, device):
    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    if cell.shape == (3,):
        cell = torch.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,) or (3, 3)")
    return cell


def _cutoff_values(distance, cutoff_radii):
    """Evaluate the polynomial cutoff for arbitrary leading distance axes."""
    distance = distance[..., None]
    cutoff_radii = cutoff_radii.reshape(*([1] * (distance.ndim - 1)), -1)
    values = (1 - distance / cutoff_radii) ** 2
    return torch.where(distance <= cutoff_radii, values, torch.zeros_like(values))


class _ACSFBase(nn.Module):
    def __init__(
        self,
        cutoff_radii,
        periodic: Sequence[bool],
        box_length,
        *,
        allow_empty=False,
    ):
        super().__init__()
        cutoff_radii = torch.as_tensor(cutoff_radii)
        if cutoff_radii.ndim != 1 or (cutoff_radii.numel() == 0 and not allow_empty):
            raise ValueError("cutoff_radii must be a non-empty one-dimensional tensor")
        if not torch.all(torch.isfinite(cutoff_radii)) or torch.any(cutoff_radii <= 0):
            raise ValueError("cutoff radii must be finite and positive")
        if len(periodic) != 3:
            raise ValueError("periodic must contain exactly three values")

        self.register_buffer("cutoff_radii", cutoff_radii)
        self.register_buffer("periodic", torch.as_tensor(periodic, dtype=torch.bool))
        self.register_buffer(
            "cell",
            _cell_tensor(
                box_length,
                dtype=cutoff_radii.dtype,
                device=cutoff_radii.device,
            ),
        )

    def _batch_cells(self, positions, cell):
        if cell is None:
            return self.cell.to(positions).expand(positions.shape[0], -1, -1)

        cell = torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
        if cell.shape == (3,):
            cell = torch.diag(cell)
        elif cell.shape != (3, 3) and cell.shape == (positions.shape[0], 3):
            cell = torch.diag_embed(cell)
        if cell.shape == (3, 3):
            cell = cell.expand(positions.shape[0], -1, -1)
        if cell.shape != (positions.shape[0], 3, 3):
            raise ValueError(
                "cell must have shape (3,), (3, 3), (batch, 3), " "or (batch, 3, 3)"
            )
        return cell

    def _edge_geometry(self, positions, cell, periodic=None, cutoff_radius=None):
        if cutoff_radius is None:
            cutoff_radius = float(self.cutoff_radii.max().detach().cpu())
        if periodic is None:
            periodic = self.periodic
        periodic = torch.as_tensor(periodic, dtype=torch.bool)
        if periodic.shape != (3,):
            raise ValueError("periodic must contain exactly three values")
        index_i, index_j, shifts = primitive_neighbor_list(
            "ijS",
            periodic.detach().cpu().numpy(),
            cell.detach().cpu().double().numpy(),
            positions.detach().cpu().double().numpy(),
            cutoff_radius,
            self_interaction=False,
        )
        index_i = torch.as_tensor(index_i, dtype=torch.long, device=positions.device)
        index_j = torch.as_tensor(index_j, dtype=torch.long, device=positions.device)
        shifts = torch.as_tensor(shifts, dtype=positions.dtype, device=positions.device)
        vectors = positions[index_j] + shifts @ cell - positions[index_i]
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        return index_i, vectors, distances

    def neighbor_geometry(
        self, positions, cell=None, periodic=None, cutoff_radius=None
    ):
        """Build reusable neighbor geometry for every structure in a batch."""
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (batch, atoms, 3)")
        cells = self._batch_cells(positions, cell)
        return [
            self._edge_geometry(
                structure_positions,
                structure_cell,
                periodic,
                cutoff_radius=cutoff_radius,
            )
            for structure_positions, structure_cell in zip(positions, cells)
        ]


class Radial(_ACSFBase):
    """Polynomial radial symmetry functions with periodic neighbor lists."""

    def __init__(
        self,
        cutoff_radii,
        periodic=(False, False, False),
        box_length=(100.0, 100.0, 100.0),
    ):
        super().__init__(cutoff_radii, periodic, box_length)

    def forward(self, positions, cell=None, periodic=None, neighbor_geometry=None):
        """Return radial descriptors with shape ``(batch, functions, atoms)``."""
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (batch, atoms, 3)")
        if neighbor_geometry is None:
            neighbor_geometry = self.neighbor_geometry(positions, cell, periodic)
        structures = []
        for index_i, _, distances in neighbor_geometry:
            values = _cutoff_values(distances, self.cutoff_radii.to(positions))
            descriptors = positions.new_zeros(
                (positions.shape[1], self.cutoff_radii.numel())
            )
            descriptors = descriptors.index_add(0, index_i, values)
            structures.append(descriptors.transpose(0, 1))
        return torch.stack(structures)


class Angular(_ACSFBase):
    """Polynomial angular symmetry functions with periodic neighbor lists."""

    def __init__(
        self,
        cutoff_radii,
        lambda_,
        zeta,
        periodic=(False, False, False),
        box_length=(100.0, 100.0, 100.0),
    ):
        super().__init__(cutoff_radii, periodic, box_length, allow_empty=True)
        lambda_ = torch.as_tensor(lambda_, dtype=self.cutoff_radii.dtype)
        zeta = torch.as_tensor(zeta, dtype=self.cutoff_radii.dtype)
        if (
            lambda_.shape != self.cutoff_radii.shape
            or zeta.shape != self.cutoff_radii.shape
        ):
            raise ValueError(
                "cutoff_radii, lambda_, and zeta must have identical shapes"
            )
        if not torch.all(torch.abs(lambda_) == 1):
            raise ValueError("lambda_ values must be +1 or -1")
        if torch.any(zeta < 1) or not torch.all(zeta == torch.round(zeta)):
            raise ValueError("zeta values must be positive integers")
        self.register_buffer("lambda_", lambda_)
        self.register_buffer("zeta", zeta)

    @property
    def n_ang_fun(self):
        """Number of configured angular symmetry functions."""
        return self.cutoff_radii.numel()

    def forward(self, positions, cell=None, periodic=None, neighbor_geometry=None):
        """Return angular descriptors with shape ``(batch, functions, atoms)``."""
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (batch, atoms, 3)")
        if self.n_ang_fun == 0:
            return positions.new_zeros((positions.shape[0], 0, positions.shape[1]))
        if neighbor_geometry is None:
            neighbor_geometry = self.neighbor_geometry(positions, cell, periodic)
        structures = []
        for structure_positions, geometry in zip(positions, neighbor_geometry):
            index_i, vectors, distances = geometry
            within_cutoff = distances <= self.cutoff_radii.max().to(positions)
            index_i = index_i[within_cutoff]
            vectors = vectors[within_cutoff]
            distances = distances[within_cutoff]
            cutoff_weights = _cutoff_values(distances, self.cutoff_radii.to(positions))
            atoms = []
            for atom_index in range(positions.shape[1]):
                atom_mask = index_i == atom_index
                atom_vectors = vectors[atom_mask]
                atom_distances = distances[atom_mask]
                atom_cutoffs = cutoff_weights[atom_mask]
                if atom_vectors.shape[0] < 2:
                    atoms.append(
                        self.cutoff_radii.to(positions) * 0
                        + structure_positions.sum() * 0
                    )
                    continue

                pair_j, pair_k = torch.triu_indices(
                    atom_vectors.shape[0],
                    atom_vectors.shape[0],
                    offset=1,
                    device=positions.device,
                )
                unit_vectors = atom_vectors / atom_distances[:, None].clamp_min(
                    torch.finfo(positions.dtype).eps
                )
                cosines = (unit_vectors[pair_j] * unit_vectors[pair_k]).sum(dim=-1)
                cosines = cosines.clamp(-1.0, 1.0)
                angular_part = (
                    1 + self.lambda_.to(positions)[:, None] * cosines[None, :]
                ).clamp_min(0.0) ** self.zeta.to(positions)[:, None]
                radial_part = atom_cutoffs[pair_j].transpose(0, 1) * atom_cutoffs[
                    pair_k
                ].transpose(0, 1)
                normalization = 2 ** (1 - self.zeta.to(positions))
                atoms.append(normalization * (angular_part * radial_part).sum(dim=1))
            structures.append(torch.stack(atoms, dim=1))
        return torch.stack(structures)


def cutoff(distance, cutoff_radii, dtype=None):
    """Evaluate the polynomial cutoff for a batched pair-distance matrix."""
    if distance.ndim != 3:
        raise ValueError("distance must have shape (batch, atoms, atoms)")
    result = _cutoff_values(
        distance, torch.as_tensor(cutoff_radii, device=distance.device)
    )
    result = result.permute(0, 3, 1, 2)
    return result.to(dtype=dtype) if dtype is not None else result
