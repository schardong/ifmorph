# coding: utf-8

import torch
from ifmorph.diff_operators import hessian, jacobian


class NoTimeWarpingLoss(torch.nn.Module):
    def __init__(
            self,
            shape_net: torch.nn.Module,
            src,
            tgt,
            spread=0.1
    ):
        super(NoTimeWarpingLoss, self).__init__()
        self.shape_net = shape_net.eval()
        self.src = src
        self.tgt = tgt
        self.spread = spread

    def TPS_energy(self, Y, X):
        hessian1 = hessian(Y, X)**2
        return hessian1.sum()

    def ID_energy(self, Y, X):
        jac = jacobian(Y.unsqueeze(0), X)[0]

        ux = jac[..., 0, 0]
        uy = jac[..., 0, 1]
        vx = jac[..., 1, 0]
        vy = jac[..., 1, 1]

        return ((ux - 1)**2 + uy**2 + vx**2 + (vy - 1)**2).sum()

    def det_energy(self, Y, X):
        jac = jacobian(Y.unsqueeze(0), X)[0]

        ux = jac[..., 0, 0]
        uy = jac[..., 0, 1]
        vx = jac[..., 1, 0]
        vy = jac[..., 1, 1]

        det = torch.abs(ux*vy - uy*vx)
        return ((ux - det)**2 + uy**2 + vx**2 + (vy - det)**2).sum()

    def rot_energy(self, Y, X):
        J = jacobian(Y.unsqueeze(0), X)[0]

        ux = J[..., 0, 0]
        uy = J[..., 0, 1]
        vx = J[..., 1, 0]
        vy = J[..., 1, 1]

        # |J^T.J - I|=0

        return ((ux**2+vx**2-1)**2 + 2*(ux*uy+vx*vy)**2 + (uy**2+vy**2-1)**2).sum()

    def C_energy(self, Y, X):
        J = jacobian(Y.unsqueeze(0), X)[0]

        ux = J[..., 0, 0]
        uy = J[..., 0, 1]
        vx = J[..., 1, 0]
        vy = J[..., 1, 1]

        det = (ux*vy - uy*vx)
        C = (ux**2+vx**2)**2 + 2*(ux*uy+vx*vy)**2 + (uy**2+vy**2)**2

        return (C/(torch.abs(det))).sum()

    def weight_decay_energy(self, model):
        reg_loss = 0.
        for param in model.parameters():
            if len(param.shape) != 1:
                reg_loss = reg_loss + (param**2).sum()
        return reg_loss

    def forward(self, coords, model):
        """
        coords: [N, 2]
        network_out: [N, 2]
        """
        M = model(coords)
        X = M["model_in"]
        Y = M["model_out"].squeeze()

        # data fitting
        y_tgt = model(self.tgt)['model_out']
        data_constraint = torch.sum((self.src - y_tgt)**2, dim=-1)

        # thin plate spline energy
        deformation_constraint = self.TPS_energy(Y, X)

        # identity constraint
        # deformation_constraint = self.ID_energy(Y, X)

        # rotational energy
        # deformation_constraint = self.rot_energy(Y,X)

        # det energy
        # deformation_constraint = self.det_energy(Y,X)

        # C energy
        # deformation_constraint = self.C_energy(Y,X) #it did not work

        # weight decay energy
        # reg_loss = self.weight_decay_energy(model)

        return {
            "data_constraint": data_constraint.mean() * 1e6,
            # "reg_loss": reg_loss,
            "deformation_constraint": deformation_constraint.mean() * 1e-2,  # id_energy 1e-1 #TPS_energy 1e-2
        }


DEFAULT_WEIGHTS_FMLOSS = {
    "data_constraint":  1e4,
    "identity_constraint":  1e3,
    "inv_constraint": 1e4,
    "TPS_constraint": 1e3,
}


class FeatureMatchingWarpingLoss(torch.nn.Module):
    """Warping loss with feature matching between source and target.

    Parameters
    ----------
    image_src: torch.nn.Module
        A network representing the source image. Must be twice differentiable.

    image_tgt: torch.nn.Module
        A network representing the target image. Must be twice differentiable.

    warp_src_pts: torch.Tensor
        An Nx2 tensor with the feature locations in the source image. Note that
        these points must be normalized to the [-1, 1] range.

    warp_tgt_pts: torch.Tensor
        An Nx2 tensor with the feature locations in the target image. Note that
        these points must be normalized to the [-1, 1] range.

    intermediate_times: list, optional
        List of intermediate times where the data constraint will be fit. All
        values must be in range [0, 1]. By default is [0.25, 0.5, 0.75]

    constraint_weights: dict, optional
        The weight of each constraint in the final loss composition. By
        default, the weights are:
        {
            "data_constraint":  1e4,
            "identity_constraint":  1e3,
            "inv_constraint": 1e4,
            "TPS_constraint": 1e3,
        }
    """
    def __init__(
            self,
            image_src: torch.nn.Module,
            image_tgt: torch.nn.Module,
            warp_src_pts: torch.Tensor,
            warp_tgt_pts: torch.Tensor,
            intermediate_times: list = [0.25, 0.5, 0.75],
            constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS
    ):
        super(FeatureMatchingWarpingLoss, self).__init__()
        self.image_src = image_src.eval()
        self.image_tgt = image_tgt.eval()
        self.src = warp_src_pts
        self.tgt = warp_tgt_pts
        self.intermediate_times = intermediate_times
        self.constraint_weights = constraint_weights
        if intermediate_times is None or not len(intermediate_times):
            self.intermediate_times = [0.25, 0.5, 0.75]
        if constraint_weights is None or not len(constraint_weights):
            self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

        # Ensuring that all necessary weights are stored.
        for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
            if k not in self.constraint_weights:
                self.constraint_weights[k] = v

    def forward(self, coords, model):
        """
        coords: torch.Tensor(shape=[N, 3])
        model: torch.nn.Module
        """
        M = model(coords)
        X = M["model_in"]
        Y = M["model_out"].squeeze()

        # thin plate spline energy
        hessian1 = hessian(Y, X)
        TPS_constraint = hessian1 ** 2

        # data fitting: f(src, 1)=tgt, f(tgt,-1)=src
        src = torch.cat((self.src, torch.ones_like(self.src[..., :1])), dim=1)
        y_src = model(src)['model_out']
        tgt = torch.cat((self.tgt, -torch.ones_like(self.tgt[..., :1])), dim=1)
        y_tgt = model(tgt)['model_out']
        data_constraint = (self.tgt - y_src)**2 + (self.src - y_tgt)**2
        data_constraint *= 1e2

        # forcing the feature matching along time
        for t in self.intermediate_times:
            tgt_0 = torch.cat((self.tgt, (t-1)*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_0 = model(tgt_0)['model_out']
            src_0 = torch.cat((self.src, t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_0 = model(src_0)['model_out']
            data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

            src_t = torch.cat((y_src_0, -t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_t = model(src_t)['model_out']
            data_constraint += ((y_src_t - self.src)**2)*2e1

            tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_t = model(tgt_t)['model_out']
            data_constraint += ((y_tgt_t - self.tgt)**2)*2e1

        # identity constraint: f(p,0) = (p)
        diff_constraint = (Y - X[..., :2])**2
        identity_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, diff_constraint, torch.zeros_like(diff_constraint))

        # inverse constraint: f(f(p,t), -t) = p,  f(f(p,-t), t) = p
        Ys = torch.cat((Y, -X[..., -1:]), dim=1)
        model_Xs = model(Ys)
        Xs = model_Xs['model_out']

        # inverse constraint: f(f(p,t), 1-t) = f(p,1)
        Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
        model_Xt = model(Yt)
        Xt = model_Xt['model_out']
        Y1 = torch.cat((X[...,0:2], torch.ones_like(X[..., -1:])), dim=1)
        X1 = model(Y1)['model_out']

        inv_constraint = (Xs - X[..., 0:2])**2 + (Xt - X1)**2
        inv_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, torch.zeros_like(inv_constraint), inv_constraint)

        return {
            "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
            "identity_constraint": identity_constraint.mean() * self.constraint_weights["identity_constraint"],
            "inv_constraint": inv_constraint.mean() * self.constraint_weights["inv_constraint"],
            "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
        }


class FeatureMatchingMorphingLoss(torch.nn.Module):
    def __init__(
            self,
            warping: torch.nn.Module,
            img_src: torch.nn.Module,
            img_tgt: torch.nn.Module,
            src_landmarks: torch.Tensor,
            tgt_landmarks: torch.Tensor
    ):
        super(FeatureMatchingMorphingLoss, self).__init__()
        self.warping = warping.eval()
        self.img_src = img_src.eval()
        self.img_tgt = img_tgt.eval()
        self.src_landmarks = src_landmarks
        self.tgt_landmarks = tgt_landmarks

    def slerp(self, val, low, high):
        low_norm = low/(torch.norm(low, dim=1, keepdim=True))
        high_norm = high/(torch.norm(high, dim=1, keepdim=True))
        omega = torch.acos((low_norm*high_norm).sum(1).clamp(-0.999, 0.999)).unsqueeze(-1)
        so = torch.sin(omega)
        # torch.where(so!=0)
        res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
        return res

    # blending in the gradient domain: interpolation of the gradients
    def forward_(self, coords, model):
        """
        coords: [N, 3]: (x,y,t)
        network_out: [N, RGB]: RGB
        """
        morphing = model(coords)
        X = morphing["model_in"]

        pt = coords.clone().detach().requires_grad_(True)
        p = pt[..., 0:2]
        t = pt[..., -1:]

        # data fitting: f(p, 0)=src(p), f(p,1)=tgt(p)
        X0 = torch.cat((p, torch.zeros_like(t)), dim=1)
        Y0_model = model(X0)
        Y0 = Y0_model['model_out']
        src_m = self.img_src(p)
        src = src_m['model_out']

        X1 = torch.cat((p, torch.ones_like(t)), dim=1)
        Y1 = model(X1)['model_out']
        tgt = self.img_tgt(p)['model_out']

        data_constraint = (src - Y0)**2 + (tgt - Y1)**2

        # (p,t) -> T(p,-t) (time zero: src)
        Sx = torch.cat((p, -t), dim=1)
        model_Xs = self.warping(Sx, preserve_grad=True)
        Xs = model_Xs['model_out']
        model_src = self.img_src(Xs, preserve_grad=True)

        # (X,t) -> T(X,1-t) (time one: tgt)
        Yt = torch.cat((p, 1 - t), dim=1)
        model_Xt = self.warping(Yt, preserve_grad=True)
        Xt = model_Xt['model_out']
        model_tgt = self.img_tgt(Xt, preserve_grad=True)

        image_blending = (1-t)*model_src['model_out'] + t*model_tgt['model_out']
        jac_blending = jacobian(image_blending.unsqueeze(0), pt)[0].clone().detach()

        jac_morphing = jacobian(morphing['model_out'].unsqueeze(0), X)[0]
        grad_constraint = ((jac_blending - jac_morphing)**2).sum(-1).sum(-1).squeeze(0)

        grad_norm = ((jac_blending)**2).sum(-1).sum(-1).squeeze(0)

        grad_constraint = torch.where(grad_norm<1e-1, ((jac_morphing)**2).sum(-1).sum(-1).squeeze(0), grad_constraint)

        grad_constraint = torch.where(torch.abs(Xs[..., 0]) < 1, grad_constraint, torch.zeros_like(grad_constraint))
        grad_constraint = torch.where(torch.abs(Xs[..., 1]) < 1, grad_constraint, torch.zeros_like(grad_constraint))
        grad_constraint = torch.where(torch.abs(Xt[..., 0]) < 1, grad_constraint, torch.zeros_like(grad_constraint))
        grad_constraint = torch.where(torch.abs(Xt[..., 1]) < 1, grad_constraint, torch.zeros_like(grad_constraint))

        return {
            "data_constraint": data_constraint.mean() * 1e4,
            "grad_constraint": grad_constraint.mean(),
        }

    def forward(self, coords, model):
        """
        coords: [N, 3]: (x,y,t)
        network_out: [N, RGB]: RGB
        """
        morphing = model(coords)
        X = morphing["model_in"]

        pt = coords.clone().detach().requires_grad_(True)
        p = pt[..., 0:2]
        t = pt[..., -1:]

        # data fitting: f(p, 0)=src(p), f(p,1)=tgt(p)
        X0 = torch.cat((p, torch.zeros_like(t)), dim=1)
        Y0_model = model(X0)
        Y0 = Y0_model['model_out']
        src_m = self.img_src(p)
        # src = src_m['model_out'][...,0].unsqueeze(-1)
        src = src_m['model_out']

        X1 = torch.cat((p, torch.ones_like(t)), dim=1)
        Y1 = model(X1)['model_out']
        # tgt = self.img_tgt(p)['model_out'][...,0].unsqueeze(-1)
        tgt = self.img_tgt(p)['model_out']

        data_constraint = (src - Y0)**2  # + (tgt - Y1)**2

        # (p,t) -> T(p,-t) (time zero: src)
        Sx = torch.cat((p, -t), dim=1)
        model_Xs = self.warping(Sx, preserve_grad=True)

        Xs = model_Xs['model_out']
        model_src = self.img_src(Xs, preserve_grad=True)
        # grad_src = gradient(model_src['model_out'], model_src['model_in']) # [N , 2, 1]

        # (X,t) -> T(X,1-t) (time one: tgt)
        Yt = torch.cat((p, 1 - t), dim=1)
        model_Xt = self.warping(Yt, preserve_grad=True)
        Xt = model_Xt['model_out']
        model_tgt = self.img_tgt(Xt, preserve_grad=True)
        # grad_tgt = gradient(model_tgt['model_out'], model_tgt['model_in']) # [N , 2, 1]

        #image_blending = (1-t)*model_src['model_out'][...,0].unsqueeze(-1) + t*model_tgt['model_out'][...,0].unsqueeze(-1)
        image_blending = (1-t)*model_src['model_out'] + t*model_tgt['model_out']
        #image_blending = src#(1-t)*model_src['model_out'][...,0].unsqueeze(-1) + t*model_tgt['model_out'][...,0].unsqueeze(-1)
        jac_blending = jacobian(image_blending.unsqueeze(0), pt)[0].clone().detach()
        #grad_blending = gradient(image_blending, pt).clone().detach()

        # embed()
        #grad_blending = self.slerp(t,grad_src,grad_tgt).detach()
        #grad_blending = ((1-t)*grad_src+t*grad_tgt).detach()

        #grad_m = gradient(morphing['model_out'], X)#[...,0:2]
        #grad_t = gradient(morphing['model_out'], X)[...,2]
        jac_morphing = jacobian(morphing['model_out'].unsqueeze(0), X)[0]
        grad_constraint = ((jac_blending - jac_morphing)**2).sum(-1).sum(-1).squeeze(0)

        #grad_constraint = ((grad_x - grad_m[..., 0:2])**2).sum(-1).unsqueeze(-1)
        # grad_constraint = ((grad_blending - grad_m)**2).sum(-1).unsqueeze(-1)
        grad_norm = ((jac_blending)**2).sum(-1).sum(-1).squeeze(0)

        grad_constraint = torch.where(grad_norm<1e-1, ((jac_morphing)**2).sum(-1).sum(-1).squeeze(0), grad_constraint)

        #align_constraint = 1 - F.cosine_similarity(grad_blending, grad_m, dim=-1)
        #align_constraint = torch.where(grad_norm<1e-1, torch.zeros_like(align_constraint), align_constraint)
        #grad_constraint = (grad_blending**2-grad_m**2).sum(-1)**2

        #grad_constraint = (image_blending - morphing['model_out'])**2
        #grad_constraint = ((jac_blending - jac_morphing)**2).sum(-1).squeeze(0)
        # grad_constraint = ((jac_blending - jac_morphing)**2).sum(-1).sum(-1).squeeze(0)
        grad_constraint = torch.where(torch.abs(Xs[..., 0]) < 0.4, grad_constraint, torch.zeros_like(grad_constraint))
        grad_constraint = torch.where(torch.abs(Xs[..., 1]) < 0.4, grad_constraint, torch.zeros_like(grad_constraint))
        # grad_constraint = torch.where(torch.abs(Xt[..., 0]) < 1, grad_constraint, torch.zeros_like(grad_constraint))
        # grad_constraint = torch.where(torch.abs(Xt[..., 1]) < 1, grad_constraint, torch.zeros_like(grad_constraint))

        blend_constraint = ((model_src['model_out'] - morphing['model_out'])**2).sum(-1)
        blend_constraint = torch.where(torch.abs(Xs[..., 0]) < 1, blend_constraint, torch.zeros_like(blend_constraint))
        blend_constraint = torch.where(torch.abs(Xs[..., 1]) < 1, blend_constraint, torch.zeros_like(blend_constraint))

        blend_constraint = torch.where(torch.maximum(torch.abs(Xs[..., 0]), torch.abs(Xs[..., 1])) < 0.4, torch.zeros_like(blend_constraint), blend_constraint)
        # blend_constraint = torch.where(torch.abs(Xs[..., 1]) < 0.5, torch.zeros_like(blend_constraint), blend_constraint)

        return {
            "data_constraint": data_constraint.mean() * 1e4,
            "grad_constraint": grad_constraint.mean() * 1e2,
            # "grad_t": (grad_t**2).mean()*1e4 ,
            # "align_constraint": align_constraint.mean()*1e1 ,
            # "laplace_constraint": laplace_constraint.mean()*1e-5,
            "blend_constraint": blend_constraint.mean() * 1e5,
        }
