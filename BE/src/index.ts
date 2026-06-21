import express from "express";
const app = express();

app.use(express.json());

app.get("/test", (req, res) => {
  res.json({ message: "Server Running" });
});
app.get("/verify", (req, res) => {
  res.json({ message: "Server Running" });
});

const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});