import express from "express";
import redis from "redis";
const app = express();

app.use(express.json());

app.get("/test", (req, res) => {
  res.json({ message: "Server Running" });
});
app.get("/verify", (req, res) => {
    //it will recive binary vid data and put it to the queue for processing and return the result
  res.json({ message: "Server Running" });
});

const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});