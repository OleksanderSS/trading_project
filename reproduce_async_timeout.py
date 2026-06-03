import asyncio
from src.utils.async_timeout import async_timeout

@async_timeout(timeout_seconds=1)
async def slow_function():
    await asyncio.sleep(2)
    return "Done"

async def main():
    result = await slow_function()
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
