#include <iostream>
#include <cstring>
#include <queue>
#define INF 0x3f3f3f3f

using namespace std;

const int maxn = 1e2 + 5;

struct Edge{
	int v, w, nxt;
}edge[maxn * 10];
// edge[i]�Ա�Ϊ�±� 

int n, m;
int head[maxn], cnt;
// head[i]�Ե�Ϊ�±꣬������һ���Ըõ�Ϊ���ı�
// cnt�����бߣ���������ߺͷ���� 
int deep[maxn];	// deep[i]�Ե�Ϊ�±꣬0����õ㲻�ܵ����t 
int now[maxn];  // now[i]�Ե�Ϊ�±꣬����õ�ĵ�ǰ��
int vis[maxn];	// ֻ��dfsȾɫʱʹ�ã��Ե�Ϊ�±꣬����õ��Ƿ�������㼯�� 

void init(){
	memset(vis, false, sizeof(vis));
	memset(head, -1, sizeof(head));
	cnt = 0;
	return;
}
void addedge(int a, int b, int w){
	// ����� 
	edge[cnt].v = b;
	edge[cnt].w = w;
	edge[cnt].nxt = head[a];	// ��һ����aΪ���ı� 
	head[a] = cnt;	// �洢���һ����aΪ���ı� 
	cnt++;
	//����� 
	edge[cnt].v = a;
	edge[cnt].w = w;	// ��Ϊ������ߣ����Է��������ҲΪw 
	edge[cnt].nxt = head[b];
	head[b] = cnt;
	cnt++;
	return;
}

bool bfs(int s, int t){
	memset(deep, 0, sizeof(deep));	// ������Ϊ���е㶼������t 
	queue<int> q;
	q.push(s);
	now[s] = head[s];	// s�ĵ�ǰ��Ӧ�������һ����sΪ���ı� 
	deep[s] = 1;
	while(q.size()){
		int u = q.front();
		q.pop();
		for(int i=head[u]; i!=-1; i=edge[i].nxt){
			int w = edge[i].w, v = edge[i].v;
			if(deep[v] || !w)
				continue;
			deep[v] = deep[u] + 1;
			now[v] = head[v];	// v�ĵ�ǰ��Ӧ�������һ����vΪ���ı�
			q.push(v);
			if(v == t)
				return true;	// ֻҪ���ҵ�����·�Ϳ����� 
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
				deep[v] = 0;	// ��v�޷���t 
			edge[i].w -= deep_rest;
			edge[i^1].w += deep_rest;
			rest -= deep_rest;
		}
	}
	now[s] = i;	// ��ǰ����Ҫ�����Ѿ���ե�ɵĻ� 
	return flow - rest;	// ���ػ�ʣ���������û�з��� 
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
			dfs(v);	// ����Ѿ�Ϊ0��˵����ƿ���ߣ�Ӧ�����ж� 
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
		int ans = solve(1, 2);	// ��1ΪԴ�㣬2Ϊ���������/��С����
		dfs(1);	// ��������ڼ���Ⱦɫ 
		for(int i=0; i<cnt; i+=2){
			int u = edge[i^1].v, v = edge[i].v;
			if(vis[u] != vis[v])
				cout<<u<<' '<<v<<endl;
		}
		cout<<endl;
	}
	return 0;
} 
