
            # Weather Agent Examples

            ttps://ai.pydantic.dev/examples/weather-agent/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/weather-agent/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/weather-agent/api/models/mistral/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/weather-agent/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/weather-agent/api/models/function/>)
    * [ pydantic_ai.models.fallback  ](https://ai.pydantic.dev/examples/weather-agent/api/models/fallback/>)
    * [ pydantic_graph  ](https://ai.pydantic.dev/examples/weather-agent/api/pydantic_graph/graph/>)
    * [ pydantic_graph.nodes  ](https://ai.pydantic.dev/examples/weather-agent/api/pydantic_graph/nodes/>)
    * [ pydantic_graph.state  ](https://ai.pydantic.dev/examples/weather-agent/api/pydantic_graph/state/>)
    * [ pydantic_graph.mermaid  ](https://ai.pydantic.dev/examples/weather-agent/api/pydantic_graph/mermaid/>)
    * [ pydantic_graph.exceptions  ](https://ai.pydantic.dev/examples/weather-agent/api/pydantic_graph/exceptions/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/weather-agent/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/weather-agent/<#example-code>)
  * [ Running the UI  ](https://ai.pydantic.dev/examples/weather-agent/<#running-the-ui>)
  * [ UI Code  ](https://ai.pydantic.dev/examples/weather-agent/<#ui-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/weather-agent/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/weather-agent/<../>)


Version
Showing documentation for the latest release [v0.0.31 2025-03-03](https://ai.pydantic.dev/examples/weather-agent/<https:/github.com/pydantic/pydantic-ai/releases/tag/v0.0.31>).
# Weather agent
Example of PydanticAI with multiple tools which the LLM needs to call in turn to answer a question.
Demonstrates:
  * [tools](https://ai.pydantic.dev/examples/weather-agent/tools/>)
  * [agent dependencies](https://ai.pydantic.dev/examples/weather-agent/dependencies/>)
  * [streaming text responses](https://ai.pydantic.dev/examples/weather-agent/results/#streaming-text>)
  * Building a [Gradio](https://ai.pydantic.dev/examples/weather-agent/<https:/www.gradio.app/>) UI for the agent


In this case the idea is a "weather" agent — the user can ask for the weather in multiple locations, the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use the `get_weather` tool to get the weather for those locations.
## Running the Example
To run this example properly, you might want to add two extra API keys **(Note if either key is missing, the code will fall back to dummy data, so they're not required)** :
  * A weather API key from [tomorrow.io](https://ai.pydantic.dev/examples/weather-agent/<https:/www.tomorrow.io/weather-api/>) set via `WEATHER_API_KEY`
  * A geocoding API key from [geocode.maps.co](https://ai.pydantic.dev/examples/weather-agent/<https:/geocode.maps.co/>) set via `GEO_API_KEY`


With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/weather-agent/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/weather-agent/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/weather-agent/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.weather_agent

```

```
uvrun-mpydantic_ai_examples.weather_agent

```

## Example Code
pydantic_ai_examples/weather_agent.py
            


            # Weather Agent

            [ Skip to content ](https://ai.pydantic.dev/examples/weather-agent/<#running-the-example>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/weather-agent/<../..> "PydanticAI")
PydanticAI 
Weather agent 
Type to start searching
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/weather-agent/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/weather-agent/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/weather-agent/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/weather-agent/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/weather-agent/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/weather-agent/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/weather-agent/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/weather-agent/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/weather-agent/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/weather-agent/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/weather-agent/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/weather-agent/tools/>)
    * [ Common Tools  ](https://ai.pydantic.dev/examples/weather-agent/common_tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/weather-agent/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/weather-agent/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/weather-agent/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/weather-agent/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/weather-agent/multi-agent-applications/>)
    * [ Graphs  ](https://ai.pydantic.dev/examples/weather-agent/graph/>)
    * [ Image and Audio Input  ](https://ai.pydantic.dev/examples/weather-agent/input/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/weather-agent/<../>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/examples/weather-agent/<../pydantic-model/>)
    * Weather agent  [ Weather agent  ](https://ai.pydantic.dev/examples/weather-agent/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/weather-agent/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/weather-agent/<#example-code>)
      * [ Running the UI  ](https://ai.pydantic.dev/examples/weather-agent/<#running-the-ui>)
      * [ UI Code  ](https://ai.pydantic.dev/examples/weather-agent/<#ui-code>)
    * [ Bank support  ](https://ai.pydantic.dev/examples/weather-agent/<../bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/examples/weather-agent/<../sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/examples/weather-agent/<../flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/examples/weather-agent/<../rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/examples/weather-agent/<../stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/examples/weather-agent/<../stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/weather-agent/<../chat-app/>)
    * [ Question Graph  ](https://ai.pydantic.dev/examples/weather-agent/<../question-graph/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/weather-agent/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/weather-agent/api/tools/>)
    * [ pydantic_ai.common_tools  ](https://ai.pydantic.dev/examples/weather-agent/api/common_tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/weather-agent/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/weather-agent/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/weather-agent/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/weather-agent/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/weather-agent/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/weather-agent/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/weather-agent/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/weather-agent/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/weather-agent/api/models/anthropic/>)
    * [ pydantic_ai.models.cohere  ](https://ai.pydantic.dev/examples/weather-agent/api/models/cohere/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/weather-agent/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](h
            


            # Weather Agent Example with Pydantic AI

            ```
from__future__import annotations as _annotations
importasyncio
importos
fromdataclassesimport dataclass
fromtypingimport Any
importlogfire
fromdevtoolsimport debug
fromhttpximport AsyncClient
frompydantic_aiimport Agent, ModelRetry, RunContext
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

@dataclass
classDeps:
  client: AsyncClient
  weather_api_key: str | None
  geo_api_key: str | None

weather_agent = Agent(
  'openai:gpt-4o',
  # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
  # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
  system_prompt=(
    'Be concise, reply with one sentence.'
    'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
    'then use the `get_weather` tool to get the weather.'
  ),
  deps_type=Deps,
  retries=2,
)

@weather_agent.tool
async defget_lat_lng(
  ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
"""Get the latitude and longitude of a location.
  Args:
    ctx: The context.
    location_description: A description of a location.
  """
  if ctx.deps.geo_api_key is None:
    # if no API key is provided, return a dummy response (London)
    return {'lat': 51.1, 'lng': -0.1}
  params = {
    'q': location_description,
    'api_key': ctx.deps.geo_api_key,
  }
  with logfire.span('calling geocode API', params=params) as span:
    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()
    span.set_attribute('response', data)
  if data:
    return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
  else:
    raise ModelRetry('Could not find the location')

@weather_agent.tool
async defget_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
"""Get the weather at a location.
  Args:
    ctx: The context.
    lat: Latitude of the location.
    lng: Longitude of the location.
  """
  if ctx.deps.weather_api_key is None:
    # if no API key is provided, return a dummy response
    return {'temperature': '21 °C', 'description': 'Sunny'}
  params = {
    'apikey': ctx.deps.weather_api_key,
    'location': f'{lat},{lng}',
    'units': 'metric',
  }
  with logfire.span('calling weather API', params=params) as span:
    r = await ctx.deps.client.get(
      'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()
    span.set_attribute('response', data)
  values = data['data']['values']
  # https://docs.tomorrow.io/reference/data-layers-weather-codes
  code_lookup = {
    1000: 'Clear, Sunny',
    1100: 'Mostly Clear',
    1101: 'Partly Cloudy',
    1102: 'Mostly Cloudy',
    1001: 'Cloudy',
    2000: 'Fog',
    2100: 'Light Fog',
    4000: 'Drizzle',
    4001: 'Rain',
    4200: 'Light Rain',
    4201: 'Heavy Rain',
    5000: 'Snow',
    5001: 'Flurries',
    5100: 'Light Snow',
    5101: 'Heavy Snow',
    6000: 'Freezing Drizzle',
    6001: 'Freezing Rain',
    6200: 'Light Freezing Rain',
    6201: 'Heavy Freezing Rain',
    7000: 'Ice Pellets',
    7101: 'Heavy Ice Pellets',
    7102: 'Light Ice Pellets',
    8000: 'Thunderstorm',
  }
  return {
    'temperature': f'{values["temperatureApparent"]:0.0f}°C',
    'description': code_lookup.get(values['weatherCode'], 'Unknown'),
  }

async defmain():
  async with AsyncClient() as client:
    # create a free API key at https://www.tomorrow.io/weather-api/
    weather_api_key = os.getenv('WEATHER_API_KEY')
    # create a free API key at https://geocode.maps.co/
    geo_api_key = os.getenv('GEO_API_KEY')
    deps = Deps(
      client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
    )
    result = await weather_agent.run(
      'What is the weather like in London and in Wiltshire?', deps=deps
    )
    debug(result)
    print('Response:', result.data)

if __name__ == '__main__':
  asyncio.run(main())

```

## Running the UI
You can build multi-turn chat applications for your agent with [Gradio](https://ai.pydantic.dev/examples/weather-agent/<https:/www.gradio.app/>), a framework for building AI web applications entirely in python. Gradio comes with built-in chat components and agent support so the entire UI will be implemented in a single python file!
Here's what the UI looks like for the weather agent:
Note, to run the UI, you'll need Python 3.10+.
```
pipinstallgradio>=5.9.0
python/uv-run-mpydantic_ai_examples.weather_agent_gradio

```

## UI Code
pydantic_ai_examples/weather_agent_gradio.py
            


            # Pydantic AI Weather Agent

            ```
importasyncio
fromdataclassesimport dataclass
fromdatetimeimport date
frompydantic_aiimport Agent
frompydantic_ai.messagesimport (
  FinalResultEvent,
  FunctionToolCallEvent,
  FunctionToolResultEvent,
  PartDeltaEvent,
  PartStartEvent,
  TextPartDelta,
  ToolCallPartDelta,
)
frompydantic_ai.toolsimport RunContext

@dataclass
classWeatherService:
  async defget_forecast(self, location: str, forecast_date: date) -> str:
    # In real code: call weather API, DB queries, etc.
    return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'
  async defget_historic_weather(self, location: str, forecast_date: date) -> str:
    # In real code: call a historical weather API or DB
    return (
      f'The weather in {location} on {forecast_date} was 18°C and partly cloudy.'
    )

weather_agent = Agent[WeatherService, str](
  'openai:gpt-4o',
  deps_type=WeatherService,
  result_type=str, # We'll produce a final answer as plain text
  system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
async defweather_forecast(
  ctx: RunContext[WeatherService],
  location: str,
  forecast_date: date,
) -> str:
  if forecast_date >= date.today():
    return await ctx.deps.get_forecast(location, forecast_date)
  else:
    return await ctx.deps.get_historic_weather(location, forecast_date)

output_messages: list[str] = []

async defmain():
  user_prompt = 'What will the weather be like in Paris on Tuesday?'
  # Begin a node-by-node, streaming iteration
  async with weather_agent.iter(user_prompt, deps=WeatherService()) as run:
    async for node in run:
      if Agent.is_user_prompt_node(node):
        # A user prompt node => The user has provided input
        output_messages.append(f'=== UserPromptNode: {node.user_prompt} ===')
      elif Agent.is_model_request_node(node):
        # A model request node => We can stream tokens from the model's request
        output_messages.append(
          '=== ModelRequestNode: streaming partial request tokens ==='
        )
        async with node.stream(run.ctx) as request_stream:
          async for event in request_stream:
            if isinstance(event, PartStartEvent):
              output_messages.append(
                f'[Request] Starting part {event.index}: {event.part!r}'
              )
            elif isinstance(event, PartDeltaEvent):
              if isinstance(event.delta, TextPartDelta):
                output_messages.append(
                  f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}'
                )
              elif isinstance(event.delta, ToolCallPartDelta):
                output_messages.append(
                  f'[Request] Part {event.index} args_delta={event.delta.args_delta}'
                )
            elif isinstance(event, FinalResultEvent):
              output_messages.append(
                f'[Result] The model produced a final result (tool_name={event.tool_name})'
              )
      elif Agent.is_call_tools_node(node):
        # A handle-response node => The model returned some data, potentially calls a tool
        output_messages.append(
          '=== CallToolsNode: streaming partial response & tool usage ==='
        )
        async with node.stream(run.ctx) as handle_stream:
          async for event in handle_stream:
            if isinstance(event, FunctionToolCallEvent):
              output_messages.append(
                f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
              )
            elif isinstance(event, FunctionToolResultEvent):
              output_messages.append(
                f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}'
              )
      elif Agent.is_end_node(node):
        assert run.result.data == node.data.data
        # Once an End node is reached, the agent run is complete
        output_messages.append(f'=== Final Agent Output: {run.result.data} ===')
            


            # Unit Testing with `TestModel`

            ### Unit testing with `TestModel`
The simplest and fastest way to exercise most of your application code is using `TestModel`[](https://ai.pydantic.dev/testing-evals/<../api/models/test/#pydantic_ai.models.test.TestModel>), this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent.
`TestModel` is not magic
The "clever" (but not too clever) part of `TestModel` is that it will attempt to generate valid structured data for [function tools](https://ai.pydantic.dev/testing-evals/<../tools/>) and [result types](https://ai.pydantic.dev/testing-evals/<../results/#structured-result-validation>) based on the schema of the registered tools.
There's no ML or AI in `TestModel`, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool.
The resulting data won't look pretty or relevant, but it should pass Pydantic's validation in most cases. If you want something more sophisticated, use `FunctionModel`[](https://ai.pydantic.dev/testing-evals/<../api/models/function/#pydantic_ai.models.function.FunctionModel>) and write your own data generation logic.
Let's write unit tests for the following application code:
weather_app.py```
importasyncio
fromdatetimeimport date
frompydantic_aiimport Agent, RunContext
fromfake_databaseimport DatabaseConn [](https://ai.pydantic.dev/testing-evals/<#__code_0_annotation_1>)
fromweather_serviceimport WeatherService [](https://ai.pydantic.dev/testing-evals/<#__code_0_annotation_2>)
weather_agent = Agent(
  'openai:gpt-4o',
  deps_type=WeatherService,
  system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
defweather_forecast(
  ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
  if forecast_date < date.today(): [](https://ai.pydantic.dev/testing-evals/<#__code_0_annotation_3>)
    return ctx.deps.get_historic_weather(location, forecast_date)
  else:
    return ctx.deps.get_forecast(location, forecast_date)

async defrun_weather_forecast( [](https://ai.pydantic.dev/testing-evals/<#__code_0_annotation_4>)
  user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
"""Run weather forecast for a list of user prompts and save."""
  async with WeatherService() as weather_service:
    async defrun_forecast(prompt: str, user_id: int):
      result = await weather_agent.run(prompt, deps=weather_service)
      await conn.store_forecast(user_id, result.data)
    # run all prompts in parallel
    await asyncio.gather(
      *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
    )

```

Here we have a function that takes a list of `(user_prompt, user_id)` tuples, gets a weather forecast for each prompt, and stores the result in the database.
**We want to test this code without having to mock certain objects or modify our code so we can pass test objects in.**
Here's how we would write tests using `TestModel`[](https://ai.pydantic.dev/testing-evals/<../api/models/test/#pydantic_ai.models.test.TestModel>):
test_weather_app.py
            