

class TheArchitect:
    def __init__(self, agents, tasks):
        # dictionary of all agents based on name 
        self.agents = {agent.name: agent for agent in agents}
        self.tasks = tasks

    def process(self, inputs):
        results = {}
        current_result = None

        for task in self.tasks:
            task_agent = self.agents[task.agent.name]
            '''to help with debugging and also checking flow of info
            we can check/print which tasks are assigned to which agent'''
            print(f"assignin task {task.name} to agent {task_agent.name}: {task.info}")

            if current_result:
                inputs['previous_result'] = current_result

            if 'search' in inputs:
                search_query = inputs['search']
                search_results = search_tool.run(query=search_query)
                inputs['search_results'] = search_results

            agent_output = task_agent.do_task(task, inputs)
            current_result = agent_output

            # send the agent's output as a message to all other agents
            for agent_name, agent in self.agents.items():
                if agent_name != task_agent.name:
                    task_agent.send_message(agent, agent_output)

            results[task.agent.name] = agent_output

        return results