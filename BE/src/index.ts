import express from "express";
import redisClient from "../infra/redisclient";

const app = express();

app.use(express.json());
app.use(express.raw({ type: "video/*", limit: "100mb" }));

app.get("/test", (_req, res) => {
  res.json({ message: "Server Running" });
});

app.post("/verify", async (req, res) => {
  const jobId = `job:${Date.now()}`;
  const videoData = req.body as Buffer;
  console.log(`[${jobId}] Request received — video size: ${videoData.length} bytes`);

  await redisClient.lPush("video:queue", JSON.stringify({
    jobId,
    data: videoData.toString("base64"),
    enqueuedAt: new Date().toISOString(),
  }));
  console.log(`[${jobId}] Job pushed to Redis queue`);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();
  console.log(`[${jobId}] SSE stream opened`);

  const send = (status: string) => {
    console.log(`[${jobId}] Sending status: ${status}`);
    res.write(`data: ${JSON.stringify({ jobId, status })}\n\n`);
  };

  send("enqueued");

  const poll = setInterval(async () => {
    const status = await redisClient.get(`${jobId}:status`);
    console.log(`[${jobId}] Polling Redis — status: ${status ?? "pending"}`);
    if (status) {
      send(status);
      if (status === "done" || status === "error") {
        console.log(`[${jobId}] Job finished with status: ${status} — closing stream`);
        clearInterval(poll);
        res.end();
      }
    }
  }, 500);

  req.on("close", () => {
    console.log(`[${jobId}] Client disconnected — clearing poll`);
    clearInterval(poll);
  });
});

const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
