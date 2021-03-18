#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <time.h>
#include <algorithm>
#include <assert.h>

#define DEBUG_TYPE 0

// 定义结构体用于存储服务器信息
struct ServerInfo
{
	// 服务器型号
	std::string serverType;
	// A点CPU核数
	int cpuCoresA;
	// A点内存大小
	int memorySizeA;
	// B点CPU核数
	int cpuCoresB;
	// B点内存大小
	int memorySizeB;
	// 购买整机消耗
	int serverCost;
	// 用户购买一天的价格
	int powerCost;
	ServerInfo() {}

	ServerInfo(std::string _serverType, int _cpuCoresA, int _memorySizeA,
				int _cpuCoresB, int _memorySizeB, int _serverCost, int _powerCost) :
	serverType(_serverType), cpuCoresA(_cpuCoresA), cpuCoresB(_cpuCoresB), 
	memorySizeA(_memorySizeA), memorySizeB(_memorySizeB),
	serverCost(_serverCost), powerCost(_powerCost) {}

	ServerInfo(const ServerInfo &rhs)
	{
		serverType = rhs.serverType;
		cpuCoresA = rhs.cpuCoresA;
		memorySizeA = rhs.memorySizeA;
		cpuCoresB = rhs.cpuCoresB;
		memorySizeB = rhs.memorySizeB;
		serverCost = rhs.serverCost;
		powerCost = rhs.powerCost;
	}
	ServerInfo& operator=(const ServerInfo &rhs)
	{
		serverType = rhs.serverType;
		cpuCoresA = rhs.cpuCoresA;
		memorySizeA = rhs.memorySizeA;
		cpuCoresB = rhs.cpuCoresB;
		memorySizeB = rhs.memorySizeB;
		serverCost = rhs.serverCost;
		powerCost = rhs.powerCost;
		return *this;
	}

	bool operator<(const ServerInfo &rhs)
	{
		int tmpCmp1 = this->serverCost / (this->cpuCoresA + this->cpuCoresB) + 
			this->serverCost / (this->memorySizeA + this->memorySizeB) +
			this->powerCost;
		int tmpCmp2 = rhs.serverCost / (rhs.cpuCoresA + rhs.cpuCoresB) + 
			rhs.serverCost / (rhs.memorySizeA + rhs.memorySizeB) +
			rhs.powerCost;
		return tmpCmp1 < tmpCmp2;
	}

	void swap(ServerInfo &rhs)
	{
		std::swap(this->serverType, rhs.serverType);
		std::swap(this->cpuCoresA, rhs.cpuCoresA);
		std::swap(this->memorySizeA, rhs.memorySizeA);
		std::swap(this->cpuCoresB, rhs.memorySizeB);
		std::swap(this->memorySizeB, rhs.memorySizeB);
		std::swap(this->serverCost, rhs.serverCost);
		std::swap(this->powerCost, rhs.powerCost);
	}
};

// 定义结构体用于存储虚拟机信息
struct VmInfo
{
	// CPU核数
	int cpuCores;
	// 内存大小
	int memorySize;
	// 是否双节点部署
	int isDouble;
};

// 定义结构体用于存储添加和删除命令
struct Command
{
	int commandType;
	std::string vmType;
	int vmId;
	Command(int _commandType, std::string _vmType, int _vmId) :
		commandType(_commandType), vmType(_vmType), vmId(_vmId) {}
};


// 定义哈希表映射服务器和虚拟机的信息
std::vector<ServerInfo> serverInfos;
std::unordered_map<std::string, VmInfo> vmInfos;

// 定义命令
std::vector<std::vector<Command> > commands;

// 每一天需要的 A 节点的资源和 B 节点的资源总数
std::vector<std::vector<int> > totalResources;

// 定义已经购买的服务器的ID与服务器的哈希表
std::unordered_map<int, ServerInfo> purchaseInfos;

// 定义虚拟机ID与虚拟机名称的哈希表
std::unordered_map<int, std::string> vmIdToType;

// 定义虚拟机ID与服务器ID对应的哈希表
std::unordered_map<int, int> vmIdToServerId;

// 定义虚拟机部署A与B节点对应的哈希表
std::unordered_map<int, int> vmIdToEnd;

// 生成服务器信息
void generateServer(std::string &serverType, std::string &cpuCores, std::string &memorySize,
    std::string &serverCost, std::string &powerCost)
{
    std::string _serverType = "";
    for (int i = 1; i < serverType.size() - 1; ++i)
	{
        _serverType += serverType[i];
    }

    int _cpuCores = 0, _memorySize = 0, _serverCost = 0, _powerCost = 0;
    for (int i = 0; i < cpuCores.size() - 1; ++i)
	{
        _cpuCores = 10 * _cpuCores + cpuCores[i] - '0';
    }

    for (int i = 0; i < memorySize.size() - 1; ++i)
	{
        _memorySize = 10 * _memorySize + memorySize[i] - '0';
    }

    for (int i = 0; i < serverCost.size() - 1; ++i)
	{
        _serverCost = 10 * _serverCost + serverCost[i] - '0';
    }

    for (int i = 0; i < powerCost.size() - 1; ++i)
	{
        _powerCost = 10 * _powerCost + powerCost[i] - '0';
    }
	// 原地构造
	ServerInfo tmpInfo(_serverType, _cpuCores / 2, _memorySize / 2,
			_cpuCores / 2, _memorySize / 2, _serverCost, _powerCost);
	serverInfos.emplace_back(tmpInfo);
}

// 生成虚拟机信息
void generateVm(std::string &vmType, std::string &cpuCores, std::string &memorySize, std::string &isDouble)
{
	std::string _vmType = "";
	// type样式 (xxxxxx,
	for (int i = 1; i < vmType.size() - 1; ++i)
	{
		_vmType += vmType[i];
	}

	int _cpuCores = 0, _memorySize = 0, _isDouble = 0;

	for (int i = 0; i < cpuCores.size() - 1; ++i)
	{
		_cpuCores = 10 * _cpuCores + cpuCores[i] - '0';
	}

	for (int i = 0; i < memorySize.size() - 1; ++i)
	{
		_memorySize = 10 * _memorySize + memorySize[i] - '0';
	}

	_isDouble = isDouble[0] - '0';

	vmInfos[_vmType] = VmInfo{_cpuCores, _memorySize, _isDouble};
}

// 生成命令信息
void generateCommand(int day, std::string &commandType, std::string &vmType, std::string &vmId)
{
	int _commandType = 0, _vmId = 0;
	std::string _vmType = "";
	if (commandType[1] == 'a')
		_commandType = 1;
	else 
		_commandType = 0;
	
	if (vmType.size())
		for (int i = 0; i < vmType.size() - 1; ++i)
		{
			_vmType += vmType[i];
		}

	for (int i = 0; i < vmId.size() - 1; ++i)
	{
		_vmId = 10 * _vmId + vmId[i] - '0';
	}
	// 添加虚拟机Id 与 Type哈希表
	if (_commandType)
		vmIdToType[_vmId] = _vmType;
	commands[day].emplace_back(_commandType, _vmType, _vmId);
}

int main()
{
	if (DEBUG_TYPE == 1)
	{
		// for debug
		std::string filePath = "./training-1.txt";
		// 重定向输入到标准输入
		std::freopen(filePath.c_str(), "rb", stdin);
	}
	// 输入服务器信息
    int serverNum;
    std::string serverType, cpuCores, memorySize, serverCost, powerCost;
    scanf("%d", &serverNum);
    for(int i = 0; i < serverNum; i++)
	{
        std::cin >> serverType >> cpuCores >> memorySize >> serverCost >> powerCost;
        generateServer(serverType, cpuCores, memorySize, serverCost, powerCost);
    }

	std::sort(serverInfos.begin(), serverInfos.end());

	// 输入虚拟机信息
	int vmNum;
	std::string vmType, isDouble;
	scanf("%d", &vmNum);
	for (int i = 0; i < vmNum; ++i)
	{
		std::cin >> vmType >> cpuCores >> memorySize >> isDouble;
		generateVm(vmType, cpuCores, memorySize, isDouble);
	}

	// 一次性读入所有天数的信息
	int dayNum, commandNum;
	std::string vmId, commandType;
	scanf("%d", &dayNum);
	// 预留空间
	commands.resize(dayNum);
	for (int i = 0; i < dayNum; ++i)
	{
		scanf("%d", &commandNum);
		// 预留空间
		commands[i].reserve(commandNum);
		for (int j = 0; j < commandNum; ++j)
		{
			std::cin >> commandType;
			// 当前命令是add
			if (commandType[1] == 'a')
			{
				std::cin >> vmType >> vmId;
			}
			else
			{
				std::cin >> vmId;
				vmType = "";
			}
			// 生成命令信息
			generateCommand(i, commandType, vmType, vmId);
		}
	}

	// TODO:process
	// 当前购买的服务器的编号
	int cnt = 0;
	std::vector<std::string> res;
	for (int i = 0; i < dayNum; ++i)
	{
		// 中间变量保存 当天购买的服务器的型号 与 对应的数量
		std::unordered_map<int, int> purchaseToday;
		// 存储中间输出结果
		std::vector<std::string> tmpRes;
		// 中间变量保存申请服务器序号的顺序
		std::vector<int> purchaseOrderToday;
		// 中间变量表示 当前申请的编号 与 服务器编号 的哈希表
		std::unordered_map<int, int> purchaseHashToday;
		int purchaseCntToday = 0;

		for (int j = 0; j < commands[i].size(); ++j)
		{
			// std::cout << purchaseInfos.size() << std::endl;
			auto tmpCommand = commands[i][j];
			// 如果是部署虚拟机命令
			if (tmpCommand.commandType == 1)
			{
				// 是否双节点部署
				int isDouble = vmInfos[tmpCommand.vmType].isDouble;
				int cpuCores = vmInfos[tmpCommand.vmType].cpuCores;
				int memorySize = vmInfos[tmpCommand.vmType].memorySize;

				bool judge = false;
				// 首先判断当前购买的机器是否有足够的空间
				// 双节点部署
				if (isDouble)
				{
					for (auto it = purchaseInfos.begin(); it != purchaseInfos.end(); ++it)
					{
						if ((it->second).cpuCoresA >= cpuCores / 2 && (it->second).memorySizeA >= memorySize / 2 &&
							(it->second).cpuCoresB >= cpuCores / 2 && (it->second).memorySizeB >= memorySize / 2)
						{
							std::string tmpStr = '(' + std::to_string(it->first) + ")\n";
							tmpRes.emplace_back(tmpStr);
							(it->second).cpuCoresA -= cpuCores / 2;
							(it->second).memorySizeA -= memorySize / 2;
							(it->second).cpuCoresB -= cpuCores / 2;
							(it->second).memorySizeB -= memorySize / 2;
							judge = true;
							vmIdToServerId[tmpCommand.vmId] = it->first;
							break;
						}
					}
					if (judge)
						continue;
					// 没有足够的服务器，需要重新购买服务器
					for (int k = 0; k < serverInfos.size(); ++k)
					{
						auto server = serverInfos[k];
						// std::cout << server.cpuCoresA << " " << server.memorySizeA << std::endl;
						if (server.cpuCoresA >= cpuCores / 2 && server.memorySizeA >= memorySize / 2 &&
							server.cpuCoresB >= cpuCores / 2 && server.memorySizeB >= memorySize / 2)
						{
							// std::cout << server.serverType << std::endl;
							// 加入到新购买服务器表中
							purchaseCntToday++;
							purchaseInfos.insert({cnt++, server});
							purchaseInfos[cnt - 1].cpuCoresA -= cpuCores / 2;
							purchaseInfos[cnt - 1].memorySizeA -= memorySize / 2;
							purchaseInfos[cnt - 1].cpuCoresB -= cpuCores / 2;
							purchaseInfos[cnt - 1].memorySizeB -= memorySize / 2;
							// 购买服务器总编号与服务器编号的哈希对应表
							purchaseHashToday[cnt - 1] = k;
							// 将当前申请的序号加入到purchaseOrderToday中
							if (find(purchaseOrderToday.begin(), purchaseOrderToday.end(), k) == purchaseOrderToday.end())
								purchaseOrderToday.push_back(k);
							if (purchaseToday.find(k) == purchaseToday.end())
							{
								purchaseToday[k] = 0;
							}
							purchaseToday[k]++;
							std::string tmpStr = '(' + std::to_string(cnt - 1) + ")\n";
							tmpRes.emplace_back(tmpStr);
							vmIdToServerId[tmpCommand.vmId] = cnt - 1;
							break;
						}
					}
				}
				else 
				{
					for (auto it = purchaseInfos.begin(); it != purchaseInfos.end(); ++it)
					{
						if ((it->second).cpuCoresA >= cpuCores && (it->second).memorySizeA >= memorySize)
						{
							std::string tmpStr = '(' + std::to_string(it->first) + ", A)\n";
							tmpRes.emplace_back(tmpStr);
							(it->second).cpuCoresA -= cpuCores;
							(it->second).memorySizeA -= memorySize;
							judge = true;
							vmIdToEnd[tmpCommand.vmId] = 1;
							vmIdToServerId[tmpCommand.vmId] = it->first;
							break;
						}
						else if ((it->second).cpuCoresB >= cpuCores && (it->second).memorySizeB >= memorySize)
						{
							std::string tmpStr = '(' + std::to_string(it->first) + ", B)\n";
							tmpRes.emplace_back(tmpStr);
							(it->second).cpuCoresB -= cpuCores;
							(it->second).memorySizeB -= memorySize;
							judge = true;
							vmIdToEnd[tmpCommand.vmId] = 2;
							vmIdToServerId[tmpCommand.vmId] = it->first;
							break;
						}
					}
					if (judge)
						continue;
					// 没有足够的服务器，需要重新购买服务器
					for (int k = 0; k < serverInfos.size(); ++k)
					{
						if (serverInfos[k].cpuCoresA >= cpuCores && serverInfos[k].memorySizeA >= memorySize)
						{
							// 加入到新购买服务器表中
							purchaseCntToday++;
							purchaseInfos.insert({cnt++, serverInfos[k]});
							purchaseInfos[cnt - 1].cpuCoresA -= cpuCores;
							purchaseInfos[cnt - 1].memorySizeA -= memorySize;
							purchaseHashToday[cnt - 1] = k;
							// 将当前申请的序号加入到purchaseOrderToday中
							if (find(purchaseOrderToday.begin(), purchaseOrderToday.end(), k) == purchaseOrderToday.end())
								purchaseOrderToday.push_back(k);
							if (purchaseToday.find(k) == purchaseToday.end())
							{
								purchaseToday[k] = 0;
							}
							purchaseToday[k]++;
							// 部署在任意一个节点上面
							std::string tmpStr = '(' + std::to_string(cnt - 1) + ", A)\n";
							tmpRes.emplace_back(tmpStr);
							vmIdToEnd[tmpCommand.vmId] = 1;
							vmIdToServerId[tmpCommand.vmId] = cnt - 1;
							break;
						}
					}
				}
			}
			// 如果是删除虚拟机操作
			else
			{
				int delVmId = tmpCommand.vmId;
				int serverId = vmIdToServerId[delVmId];
				// 恢复对应的服务器的资源
				auto tmpVmInfo = vmInfos[vmIdToType[delVmId]];
				// 删除双节点部署的虚拟机
				if (tmpVmInfo.isDouble)
				{
					purchaseInfos[serverId].cpuCoresA += tmpVmInfo.cpuCores / 2;
					purchaseInfos[serverId].memorySizeA += tmpVmInfo.memorySize / 2;
					purchaseInfos[serverId].cpuCoresB += tmpVmInfo.cpuCores / 2;
					purchaseInfos[serverId].memorySizeB += tmpVmInfo.memorySize / 2;
					vmIdToType.erase(delVmId);
				}
				// 删除单节点部署的虚拟机
				else
				{
					// 在A点部署
					if (vmIdToEnd[delVmId] == 1)
					{
						purchaseInfos[serverId].cpuCoresA += tmpVmInfo.cpuCores;
						purchaseInfos[serverId].memorySizeA += tmpVmInfo.memorySize;
					}
					// 在B点部署
					else 
					{
						purchaseInfos[serverId].cpuCoresB += tmpVmInfo.cpuCores;
						purchaseInfos[serverId].memorySizeB += tmpVmInfo.memorySize;
					}
					vmIdToType.erase(delVmId);
					vmIdToEnd.erase(delVmId);
				}
			}
		}

		// 重新整理一下编号顺序信息
		// 当天的起始编号
		int start_index = purchaseInfos.size() - purchaseCntToday;
		int cur_cnt = purchaseInfos.size() - purchaseCntToday;
		std::unordered_map<int, ServerInfo> tmpServerInfos;
		std::unordered_map<int, int> numberHash;
		for (int k = 0; k < purchaseOrderToday.size(); ++k)
		{
			for (int j = 0; j < purchaseHashToday.size(); ++j)
			{
				auto tmpServerNumber = purchaseHashToday[j + start_index];
				if (tmpServerNumber == purchaseOrderToday[k])
				{
					numberHash[j + start_index] = cur_cnt;
					// 将对应的信息进行交换
					tmpServerInfos.insert({cur_cnt, purchaseInfos[j + start_index]});
					cur_cnt++;
				}
			}
		}
		for (int j = 0; j < purchaseHashToday.size(); ++j)
		{
			purchaseInfos[j + start_index] = tmpServerInfos[j + start_index];
		}

		// 调整虚拟机对应的服务器ID
		for (auto it = vmIdToServerId.begin(); it != vmIdToServerId.end(); ++it)
		{
			// 在start_index之前的就不用调整了
			if (it->second < start_index)
				continue;
			it->second = numberHash[it->second];
		}

		// 调整输出信息
		std::vector<std::string> resForRequest;
		for (auto &s : tmpRes)
		{
			int tmpNum = 0;
			int j = 1;
			for (j = 1; j < s.size(); ++j)
			{
				if (s[j] >= '0' && s[j] <= '9')
					tmpNum = tmpNum * 10 + s[j] - '0';
				else break;
			}
			// 如果编号小于本次开始的编号, 则不用调整
			if (tmpNum < start_index)
			{
				resForRequest.emplace_back(s);
				continue;
			}
			std::string tmpStr = '(' + std::to_string(numberHash[tmpNum]) + s.substr(j, s.size() - j);
			resForRequest.emplace_back(tmpStr);
		}

		// 整理当天的购买信息
		res.emplace_back("(purchase, " + std::to_string(purchaseOrderToday.size()) + ")\n");
		for (int j = 0; j < purchaseOrderToday.size(); ++j)
		{
			std::string tmpStr = "";
			tmpStr += '(';
			tmpStr += serverInfos[purchaseOrderToday[j]].serverType;
			tmpStr += ", ";
			tmpStr += std::to_string(purchaseToday[purchaseOrderToday[j]]);
			tmpStr += ")\n";
			res.emplace_back(tmpStr);
		}
		res.emplace_back("(migration, 0)\n");

		// 整理当天的请求输出信息
		for (auto &s : resForRequest)
		{
			res.emplace_back(s);
		}
	}
	// TODO:write standard output
	if (DEBUG_TYPE == 1)
	{
		// for debug
		std::string outputPath = "./output_2_back.txt";
		// TODO:read standard input
		// 重定向输入到标准输入
		std::freopen(outputPath.c_str(), "wb", stdout);
	}
	for (auto &s : res)
	{
		std::cout << s;
	}
	// TODO:fflush(stdout);
	fflush(stdout);

	return 0;
}
