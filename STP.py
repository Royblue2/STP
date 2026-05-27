import numpy as np
import torch, gym, math





class PrioritizedReplayBuffer():
    def __init__(self, state_dim, action_dim, ip,k,max_size=int(1e6)):
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.tree = SumTree(max_size)
        self.kmeans = KMeans(k,ip=ip) ##
        self.ip = ip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if self.ptr >= 25e3:
            if self.ptr == 25e3:
                group_counts, marked = self.kmeans.fit(self.state[0:self.ptr], self.action[0:self.ptr])
                density = torch.zeros(self.ptr, dtype=torch.int, device=self.device)
                for i in range(self.ptr):
                    density[i] = group_counts[marked[i]-1]
                avg_density = torch.tensor(self.ptr, dtype=torch.float, device=self.device) / self.kmeans.k
                values = (avg_density / density).clip(1 - self.ip, 1 + self.ip)
                priorities = torch.exp(values) + torch.tensor(1, dtype=torch.float, device=self.device)
                priorities = priorities*torch.tensor(0.5, dtype=torch.float, device=self.device)
                self.levels = self.tree.batch_set(self.ptr, priorities)
            else:
                value = self.kmeans.update(self.ptr, state, action)
                priority1 = (self.levels / self.ptr)*torch.exp(value)
                priority2 = torch.tensor(self.ptr/25e3, dtype=torch.float, device=self.device)**2
                priority = (priority1 + priority2) * torch.tensor(0.5, dtype=torch.float, device=self.device)
                self.levels = self.tree.set(self.ptr, priority)

    def sample(self, batch_size):
        ind = self.tree.sample(batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device))



class SumTree(object):
    def __init__(self, max_size):
        self.levels = [np.zeros(1)]
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    def sample(self, batch_size):
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        ind = np.zeros(batch_size, dtype=int)
        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]
            is_greater = np.greater(value, left_sum)
            ind += is_greater
            value -= left_sum * is_greater
        return ind

    def set(self, ind, priority):
        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority.cpu())
            ind //= 2
        return self.levels[0][0]


    def batch_set(self, ind, priority):
        if isinstance(priority, torch.Tensor):
            priority = priority.cpu().numpy()
        batch_ind = np.arange(ind)
        for nodes in self.levels[::-1]:
            np.add.at(nodes, batch_ind, priority)
            batch_ind //= 2
        return self.levels[0][0]


class KMeans:
    def __init__(self, k,ip, max_iters=30, tol=1e-4, device=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.representatives = None
        self.group_counts = None
        self.ip = ip
        print(ip)
    def fit(self, states, actions):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        states = states.to(self.device)
        actions = actions.to(self.device)
        state_actions = torch.cat([states, actions], dim=1)  # [N, state_dim + action_dim]

        # 获取数据维度：N为样本数，D为特征维度
        N, D = state_actions.shape

        # K-means++初始化
        centroids = torch.empty(self.k, D, device=self.device, dtype=state_actions.dtype)
        first_idx = torch.randint(0, N, (1,), device=self.device)
        centroids[0] = state_actions[first_idx]
        closest_dist_sq = torch.full((N,), float('inf'), device=self.device)
        for c in range(1, self.k):
            dist_sq = torch.sum((state_actions - centroids[c - 1]) ** 2, dim=1)
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)
            probs = closest_dist_sq / closest_dist_sq.sum()
            next_idx = torch.multinomial(probs, 1)  
            centroids[c] = state_actions[next_idx]

        # K-means迭代优化
        for i in range(self.max_iters):
            distances = torch.cdist(state_actions, centroids, p=2)  # [N, k]
            labels = torch.argmin(distances, dim=1)  # [N]
            new_centroids = torch.zeros_like(centroids)
            for j in range(self.k):
                mask = (labels == j)
                if mask.sum() > 0:
                    new_centroids[j] = state_actions[mask].mean(dim=0)
                else:
                    new_centroids[j] = centroids[j]
            centroids = new_centroids

        # 为每个聚类选择最接近质心的点作为代表
        representatives = torch.zeros_like(centroids)
        for j in range(self.k):
            mask = (labels == j)
            if mask.sum() > 0:
                cluster_points = state_actions[mask]
                dist_to_centroid = torch.cdist(cluster_points, centroids[j].unsqueeze(0), p=2).squeeze(1)
                closest_idx = torch.argmin(dist_to_centroid)
                representatives[j] = cluster_points[closest_idx]
        self.representatives = representatives

        # 计算阈值
        distance_matrix = torch.cdist(self.representatives, self.representatives, p=2)
        k = self.representatives.shape[0]
        temp = torch.eye(k).to(self.device) * torch.tensor(1e6).to(self.device)
        distance_matrix = distance_matrix + temp
        distance_matrix_min = torch.min(distance_matrix, dim=1).values
        self.threshold = torch.mean(distance_matrix_min)

        # 根据阈值标记每个数据点所属的代表点
        marked = torch.zeros(N, dtype=torch.int, device=self.device)
        self.group_counts = torch.zeros(self.representatives.shape[0], device=self.representatives.device)
        for i in range(N):
            unmarked_points = state_actions[i]
            dists = torch.cdist(unmarked_points.unsqueeze(0), self.representatives)
            values, indices = torch.topk(dists, 1, largest=False)
            if values > self.threshold:
                self.k += 1
                marked[i] = self.k
                self.representatives = torch.cat([self.representatives, unmarked_points.unsqueeze(0)], dim=0)
                self.group_counts = torch.cat([self.group_counts, torch.tensor([1.0], device=self.device)])
            else:
                self.group_counts[indices] += 1
                marked[i] = indices + torch.tensor([1.0], device=self.device)
        print(self.k)
        return self.group_counts, marked

    def update(self, total, state, action):
        if isinstance(action, np.ndarray):
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
        state = state.to(self.device).unsqueeze(0)
        action = action.to(self.device).unsqueeze(0)
        state_action = torch.cat([state, action], dim=1)

        distances_list = torch.cdist(state_action, self.representatives, p=2).squeeze(0)
        min_distance = torch.min(distances_list)

        if min_distance > self.threshold:
            self.k += 1
            self.representatives = torch.cat([self.representatives, state_action], dim=0)
            self.group_counts = torch.cat([self.group_counts, torch.ones(1, device=self.representatives.device)], dim=0)
            count = torch.tensor(float(1), device=self.device)
        else:
            nearest_idx = torch.argmin(distances_list)
            self.group_counts[nearest_idx] += 1
            count = self.group_counts[nearest_idx]

        t = torch.tensor(float(total), device=self.device)
        value = (t / (self.k * count)).clip(1-self.ip, 1+self.ip)

        return value
