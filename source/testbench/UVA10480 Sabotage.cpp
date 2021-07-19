#include <iostream>
#include <cstring>
#include <queue>
#define INF 0x3f3f3f3f

using namespace std;

const int maxn = 1e2 + 5;

struct Edge{
	int v, w, nxt;
}edge[maxn * 10];
// edge[i]以边为下标 

int n, m;
int head[maxn], cnt;
// head[i]以点为下标，存放最后一条以该点为起点的边
// cnt是所有边，包括正向边和反向边 
int deep[maxn];	// deep[i]以点为下标，0代表该点不能到达点t 
int now[maxn];  // now[i]以点为下标，代表该点的当前弧
int vis[maxn];	// 只在dfs染色时使用，以点为下标，代表该点是否属于起点集合 

void init(){
	memset(vis, false, sizeof(vis));
	memset(head, -1, sizeof(head));
	cnt = 0;
	return;
}
void addedge(int a, int b, int w){
	// 正向边 
	edge[cnt].v = b;
	edge[cnt].w = w;
	edge[cnt].nxt = head[a];	// 上一条以a为起点的边 
	head[a] = cnt;	// 存储最后一条以a为起点的边 
	cnt++;
	//反向边 
	edge[cnt].v = a;
	edge[cnt].w = w;	// 因为是无向边，所以反向边容量也为w 
	edge[cnt].nxt = head[b];
	head[b] = cnt;
	cnt++;
	return;
}

bool bfs(int s, int t){
	memset(deep, 0, sizeof(deep));	// 首先认为所有点都到不了t 
	queue<int> q;
	q.push(s);
	now[s] = head[s];	// s的当前弧应当是最后一条以s为起点的边 
	deep[s] = 1;
	while(q.size()){
		int u = q.front();
		q.pop();
		for(int i=head[u]; i!=-1; i=edge[i].nxt){
			int w = edge[i].w, v = edge[i].v;
			if(deep[v] || !w)
				continue;
			deep[v] = deep[u] + 1;
			now[v] = head[v];	// v的当前弧应当是最后一条以v为起点的边
			q.push(v);
			if(v == t)
				return true;	// 只要能找到增广路就可以了 
		}
	}
	return false;
}

int dinic(int s, int t, int flow){
	if(s == t)
		return flow;
	int i, rest=flow;
	for(i=now[s]; i!=-1 && rest; i=edge[i].nxt){
		int w = edge[i].w, v = edge[i].v;
		if(w && deep[v] == deep[s] + 1){
			int deep_rest = dinic(v, t, min(rest, w));
			if(!deep_rest)
				deep[v] = 0;	// 从v无法到t 
			edge[i].w -= deep_rest;
			edge[i^1].w += deep_rest;
			rest -= deep_rest;
		}
	}
	now[s] = i;	// 当前弧需要跳过已经被榨干的弧 
	return flow - rest;	// 返回还剩余多少流量没有分配 
} 

int solve(int s, int t){
	int ans = 0;
	while(bfs(s, t)){
		int flow;
		while(flow = dinic(s, t, INF)){
			ans += flow;
		}
	}
	return ans;
}

void dfs(int s){
	vis[s] = true;
	for(int i=head[s]; i!=-1; i=edge[i].nxt){
		int v = edge[i].v;
		if(vis[v])
			continue;
		if(edge[i].w)
			dfs(v);	// 如果已经为0，说明是瓶颈边，应当被切断 
	}
	return;
} 

int main(){
	while(cin>>n>>m && n && m){
		init();
		for (int i=1;i<=m;i++){
			int a, b, w;
			cin>>a>>b>>w;
			addedge(a, b, w);
		}
		int ans = solve(1, 2);	// 求1为源点，2为汇点的最大流/最小割结果
		dfs(1);	// 对起点所在集合染色 
		for(int i=0; i<cnt; i+=2){
			int u = edge[i^1].v, v = edge[i].v;
			if(vis[u] != vis[v])
				cout<<u<<' '<<v<<endl;
		}
		cout<<endl;
	}
	return 0;
} 
