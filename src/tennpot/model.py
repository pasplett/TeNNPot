import torch
import torch.nn as nn

from tennpot.acsf import Angular, Radial
from tennpot.activations import get_activation


class TeNNPot(nn.Module):
    """Base class for the Te-depenedent neural network interatomic potential (TeNNPot)
    based on the high-dimensional neural network architecture.

    Attributes
    ----------
    radial : Radial
        The PyTorch model for calculating the radial symmetry functions using the
        specified parameters.
    angular : Angular
        The PyTorch model for calculating the angular symmetry functions using the
        specified parameters.
    atomic_nn : AtomicNN
        The PyTorch model of the atomic neural network, a multi-layer perceptron
        projecting the symmetry function values of a single atom and the electronic
        temperature to the atomic contribution to the total potential energy of the
        system.
    scale_factors : torch.Tensor
        The symmetry function values are divided by these scaling factors before
        entering the atomic neural network. This tensor has the shape (N_SYM_FUN, ).

    References
    ----------
    * Plettenberg, P., Bauerhenne B., Garcia M. E.: "Neural network interatomic
      potential for laser-excited materials". Communications Materials 4, 63 (2023).
    * Behler, J. & Parrinello, M.: "Generalized neural-network representation
      of high-dimensional potential-energy surfaces." Phys. Rev. Lett. 98, 146401
      (2007).
    """

    def __init__(
        self,
        rad_cutoff_radii,
        ang_cutoff_radii,
        lambda_,
        zeta,
        periodic=(False, False, False),
        box_length=(100.0, 100.0, 100.0),
        nodes_hid=(),
        activations=None,
        scale_factors=None,
        device=None,
        dtype=None,
    ):
        """Initializes a Te-dependent neural network interatomic potential.

        Parameters
        ----------
        rad_cutoff_radii : torch.Tensor
            Tensor containing the cutoff radii for the radial symmetry functions.
            It has the shape (N_RAD_FUN, ).
        ang_cutoff_radii : torch.Tensor
            Tensor containing the cutoff radii for the angular symmetry functions.
            It has the shape (N_ANG_FUN, ).
        lambda_ : torch.Tensor
            Tensor containing the lambda parameter for the angular symmetry functions.
            It has the shape (N_ANG_FUN, ).
        zeta : torch.Tensor
            Tensor containing the zeta parameter for the angular symmetry functions.
            It has the shape (N_ANG_FUN, ).
        periodic : list of bool, optional
            List or tuple with three boolean entries indicating whether to take into
            account periodic boundary conditions in x-, y- and z-direction. The default
            is [False, False, False].
        box_length : list of float, optional
            List or tuple with three float entries determining the size of the
            simulation cell in x-, y-, and z-direction. Required to take into account
            periodic boundary conditions. The default is [100., 100., 100.].
        nodes_hid : list of int, optional
            List containing the number of hidden nodes per hidden layer for the atomic
            neural network. The length of the list determines the number of hidden
            layers. The default is [].
        activations : list of str, optional
            List of string identifiers for the activation functions in each hidden layer
            of the atomic neural network. The default is None.
        scale_factors : torch.Tensor, optional
            The symmetry function values are divided by these scaling factors before
            entering the atomic neural network. This tensor has the shape (N_SYM_FUN, ).
            If None, no scaling is applied. The default is None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.radial = Radial(
            cutoff_radii=rad_cutoff_radii, periodic=periodic, box_length=box_length
        )
        self.angular = Angular(
            cutoff_radii=ang_cutoff_radii,
            lambda_=lambda_,
            zeta=zeta,
            periodic=periodic,
            box_length=box_length,
        )
        if scale_factors is not None:
            scale_factors = torch.as_tensor(
                scale_factors,
                device=device or self.radial.cutoff_radii.device,
                dtype=dtype or self.radial.cutoff_radii.dtype,
            )
        self.register_buffer("scale_factors", scale_factors)

        n_rad_fun = rad_cutoff_radii.shape[0]
        n_ang_fun = ang_cutoff_radii.shape[0]
        n_sym_fun = n_rad_fun + n_ang_fun

        self.atomic_nn = AtomicNN(
            n_sym_fun, nodes_hid, activations=activations, **factory_kwargs
        )

    def forward(
        self,
        positions,
        el_temp,
        cell=None,
        periodic=None,
        compute_forces=True,
        create_graph=False,
    ):
        """Forward call of the Te-dependent neural network interatomic potential.

        Parameters
        ----------
        positions : torch.Tensor
            The atomic positions in a tensor of the shape (B, N_ATOMS, 3), where B is
            the batch size and N_ATOMS is the number of atoms.
        el_temp : torch.Tensor
            Tensor containing the electronic temperatures in 1e-1 Ha (0.5 = 50 mHa,
            1 = 100 mHa). The shape of the tensor is (B, 1), where B is the batch size.
        cell : torch.Tensor, optional
            Override the default cell with a (3, 3) matrix or batched matrices.
        periodic : sequence of bool, optional
            Override the default periodicity along the three cell directions.
        compute_forces : bool, optional
            Return energy and forces when True (default), or only energy when False.
        create_graph : bool, optional
            Retain the force gradient graph for training when True (default: False).

        Returns
        -------
        phi_total : torch.Tensor
            The cohesive free energy of the system, a tensor of shape (B, 1),
            where B is the batch size.
        forces : torch.Tensor
            Returned only when compute_forces=True. Negative position gradients
            of the free energy, with shape (B, N_ATOMS, 3).
        """
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (batch, atoms, 3)")
        if el_temp.ndim != 2 or el_temp.shape != (positions.shape[0], 1):
            raise ValueError("el_temp must have shape (batch, 1)")
        if compute_forces and not positions.requires_grad:
            positions = positions.clone().requires_grad_(True)

        sym_fun = self.symmetry_functions(
            positions, cell=cell, periodic=periodic, scale=True
        )

        atomic_inputs = sym_fun.transpose(1, 2)
        atomic_temperatures = el_temp[:, None, :].expand(-1, positions.shape[1], -1)
        phi_total = self.atomic_nn(
            sym_fun=atomic_inputs,
            el_temp=atomic_temperatures,
        ).sum(dim=1)

        if not compute_forces:
            return phi_total

        (energy_gradient,) = torch.autograd.grad(
            phi_total.sum(),
            positions,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        forces = -energy_gradient
        return phi_total, forces

    def symmetry_functions(self, positions, cell=None, periodic=None, scale=True):
        """Evaluate radial and angular descriptors for a batch of structures."""
        cutoff_radius = float(self.radial.cutoff_radii.max().detach().cpu())
        if self.angular.n_ang_fun:
            cutoff_radius = max(
                cutoff_radius,
                float(self.angular.cutoff_radii.max().detach().cpu()),
            )
        geometry = self.radial.neighbor_geometry(
            positions,
            cell=cell,
            periodic=periodic,
            cutoff_radius=cutoff_radius,
        )
        sym_fun = torch.concat(
            (
                self.radial(
                    positions,
                    cell=cell,
                    periodic=periodic,
                    neighbor_geometry=geometry,
                ),
                self.angular(
                    positions,
                    cell=cell,
                    periodic=periodic,
                    neighbor_geometry=geometry,
                ),
            ),
            dim=1,
        )
        if scale and self.scale_factors is not None:
            sym_fun = self._scale(sym_fun)
        return sym_fun

    def _scale(self, sym_fun):
        """Divides the symmetry functions by the specified scaling factors."""
        return sym_fun / self.scale_factors[None, :, None]


class AtomicNN(nn.Module):
    """Base class for the atomic neural network as part of the Te-dependent
    neural network interatomic potential. It predicts the atomic energy from
    a set of symmetry functions and the electronic temperature Te.

    Attributes
    ----------
    nodes_in : int
        The number of input nodes to the atomic neural network, corresponding to the
        number of symmetry functions plus one node for the electronic temperature.
    nodes_hid : list of int
        List containing the number of nodes per hidden layer for the atomic neural
        network. The length of the list determines the number of hidden layers.
    layers : list of nn.Linear
        List containing the input and hidden network layers as nn.Linear objects.
    activations : list of nn.Module
        List of activation functions corresponding to each network layer in layers.
    out_layer : nn.Linear
        Linear output layer of the atomic neural network. In this layer, no activation
        function is applied.

    References
    ----------
    * Plettenberg, P., Bauerhenne B., Garcia M. E.: "Neural network interatomic
      potential for laser-excited materials". Comm. Mater. 4, 63 (2023).
    * Behler, J. & Parrinello, M.: "Generalized neural-network representation
      of high-dimensional potential-energy surfaces." Phys. Rev. Lett. 98, 146401
      (2007).
    """

    def __init__(
        self, n_sym_fun, nodes_hid=(), activations=None, device=None, dtype=None
    ):
        """Initializes an atomic neural network.

        Parameters
        ----------
        n_sym_fun : int
            The number of symmetry functions (radial + angular) describing the atomic
            environment. The number of input nodes to the atomic neural network
            corresponds to n_sym_fun plus one node for the electronic temperature.
        nodes_hid : list of int, optional
            List containing the number of hidden nodes per hidden layer for the atomic
            neural network. The length of the list determines the number of hidden
            layers. The default is [].
        activations : list of str, optional
            List of string identifiers for the activation functions in each hidden layer
            of the atomic neural network. If None, no activation functions are applied.
            The default is None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nodes_in = n_sym_fun + 1
        self.nodes_hid = list(nodes_hid)

        # Stack layers
        self.layers = nn.ModuleList()
        in_ = self.nodes_in
        for out_ in self.nodes_hid:
            self.layers.append(nn.Linear(in_, out_, bias=False, **factory_kwargs))
            in_ = out_
        self.out_layer = nn.Linear(in_, 1, bias=False, **factory_kwargs)
        for layer in [*self.layers, self.out_layer]:
            nn.init.xavier_normal_(layer.weight)

        # Define activation functions
        if activations is None:
            self.activations = nn.ModuleList(nn.Identity() for _ in self.nodes_hid)
        else:
            if len(activations) != len(self.nodes_hid):
                raise ValueError("activations must contain one entry per hidden layer")
            self.activations = nn.ModuleList(get_activation(act) for act in activations)

    def forward(self, sym_fun, el_temp, el_temp_bias=True):
        """Forward call of atomic neural network. Evaluates the neural network
        for a given set of symmetry functions and an electronic temperature Te.

        Parameters
        ----------
        sym_fun : torch.Tensor
            Tensor of symmetry functions (radial and angular). The shape of the tensor
            has to be (..., N_SYM_FUN), with arbitrary leading batch and atom
            dimensions.
        el_temp : torch.Tensor
            Electronic temperatures in 1e-1 Ha (0.5 = 50 mHa, 1 = 100 mHa),
            with shape (..., 1) matching the descriptor leading dimensions.
        el_temp_bias : bool
            Whether or not to subtract the Te-bias correction from the atomic potential
            energy. The Te-bias correction corresponds to the atomic NN evaluated for
            zero symmetry functions. This should always be set to True when calling
            the atomic NN. It is only set to False, when this function calls itself.
        """
        x = torch.concat((sym_fun, el_temp), dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activations[i](x)

        y = self.out_layer(x)

        if el_temp_bias:
            # Subtract Te-bias correction
            y -= self.forward(
                sym_fun=torch.zeros_like(sym_fun, requires_grad=False),
                el_temp=el_temp,
                el_temp_bias=False,
            )

        return y
