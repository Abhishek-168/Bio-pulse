import redis from "redis";

const client = redis.createClient({
    url: "redis://localhost:6379",
    legacyMode: true,
});

client.on("error", (err) => {
    console.error("Redis Client Error", err);
});

export default client;