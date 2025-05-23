import torch
import copy


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                # pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                # finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                pretrained_state_dict = pretrained_checkpoint.state_dict()
                finetuned_state_dict = finetuned_checkpoint.state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)
    
    def __mul__(self, scalar):        
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * scalar
        return TaskVector(vector=new_vector)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            # pretrained_model = torch.load(pretrained_checkpoint)
            pretrained_model = copy.deepcopy(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model



    @staticmethod
    def topk_values_mask(M, K=0.7, return_mask=False):
        if K > 1:
            K /= 100

        original_shape = M.shape
        if M.dim() == 1:
            M = M.unsqueeze(0)

        M = M.to(torch.float32)
        n, d = M.shape
        k = int(d * K)
        k = d - k

        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
        mask = M.abs() >= kth_values
        final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

        if return_mask:
            return M * final_mask, final_mask.float().mean(dim=1), final_mask
        return M * final_mask, final_mask.float().mean(dim=1)

    @staticmethod
    def resolve_zero_signs(sign_to_mult, method="majority"):
        majority_sign = torch.sign(sign_to_mult.sum())
        if method == "majority":
            sign_to_mult[sign_to_mult == 0] = majority_sign
        elif method == "minority":
            sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
        return sign_to_mult

    @staticmethod
    def resolve_sign(Tensor):
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
        sign_to_mult = TaskVector.resolve_zero_signs(sign_to_mult, "majority")
        return sign_to_mult

    @staticmethod
    def disjoint_merge(Tensor, merge_func, sign_to_mult):
        merge_func = merge_func.split("-")[-1]
        if sign_to_mult is not None:
            rows_to_keep = torch.where(
                sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
            )
            selected_entries = Tensor * rows_to_keep
        else:
            rows_to_keep = Tensor != 0
            selected_entries = Tensor * rows_to_keep

        if merge_func == "mean":
            non_zero_counts = (selected_entries != 0).sum(dim=0).float()
            disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
                non_zero_counts, min=1
            )
        elif merge_func == "sum":
            disjoint_aggs = torch.sum(selected_entries, dim=0)
        elif merge_func == "max":
            disjoint_aggs = selected_entries.abs().max(dim=0)[0]
            disjoint_aggs *= sign_to_mult
        else:
            raise ValueError(f"Merge method {merge_func} is not defined.")

        return disjoint_aggs

    @staticmethod
    def ties_merging(flat_task_checks, reset_thresh=None, merge_func=""):
        all_checks = flat_task_checks.clone()
        updated_checks, *_ = TaskVector.topk_values_mask(
            all_checks, K=reset_thresh, return_mask=False
        )
        print(f"RESOLVING SIGN")
        final_signs = TaskVector.resolve_sign(updated_checks)
        assert final_signs is not None
        
        print(f"Disjoint AGGREGATION: {merge_func}")
        merged_tv = TaskVector.disjoint_merge(updated_checks, merge_func, final_signs)
        
        return merged_tv
