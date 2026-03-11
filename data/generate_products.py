"""
Generate synthetic product data and user interaction data.
Produces:
  - data/products.csv       (5000 products)
  - data/user_interactions.csv (5000 interactions)
"""

import csv
import json
import os
import random
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

CATEGORIES = ["Electronics", "Clothing", "Footwear", "Books", "Sports", "Home", "Beauty", "Toys"]

BRANDS_BY_CATEGORY = {
    "Electronics": ["Samsung", "Apple", "Sony", "Dell", "HP", "Lenovo", "LG", "Bose", "JBL", "OnePlus", "Xiaomi", "Asus"],
    "Clothing": ["Levi's", "H&M", "Zara", "Allen Solly", "Peter England", "Van Heusen", "Raymond", "US Polo", "Tommy Hilfiger", "UCB"],
    "Footwear": ["Nike", "Adidas", "Puma", "Reebok", "Skechers", "Woodland", "Bata", "Liberty", "New Balance", "Asics"],
    "Books": ["Penguin", "HarperCollins", "Oxford", "Pearson", "McGraw Hill", "Wiley", "Cambridge", "Springer"],
    "Sports": ["Yonex", "Cosco", "Nivia", "SG", "Adidas", "Nike", "Under Armour", "Decathlon", "Wilson", "Head"],
    "Home": ["Prestige", "Philips", "Bajaj", "Havells", "Ikea", "HomeTown", "Godrej", "Wipro", "Crompton", "Usha"],
    "Beauty": ["Lakme", "Maybelline", "L'Oreal", "Nivea", "Himalaya", "Biotique", "Dove", "Neutrogena", "Garnier", "Olay"],
    "Toys": ["Lego", "Hasbro", "Mattel", "Fisher-Price", "Funskool", "Hot Wheels", "Nerf", "Barbie", "Play-Doh", "Melissa & Doug"],
}

PRODUCT_TEMPLATES = {
    "Electronics": [
        "{brand} Wireless Bluetooth Headphones", "{brand} Smart LED TV 43-inch", "{brand} Laptop {adj}",
        "{brand} Smartwatch with AMOLED Display", "{brand} Portable Bluetooth Speaker",
        "{brand} Wireless Earbuds Pro", "{brand} Gaming Monitor 27-inch", "{brand} Tablet 10-inch",
        "{brand} Mechanical Keyboard RGB", "{brand} Webcam 4K Ultra HD",
        "{brand} Power Bank 20000mAh", "{brand} External SSD 1TB",
    ],
    "Clothing": [
        "{brand} Men's Casual Shirt", "{brand} Women's Summer Dress", "{brand} Slim Fit Jeans",
        "{brand} Cotton Polo T-Shirt", "{brand} Formal Blazer", "{brand} Hooded Sweatshirt",
        "{brand} Chino Trousers", "{brand} Printed Kurta Set", "{brand} Denim Jacket",
        "{brand} Winter Wool Sweater",
    ],
    "Footwear": [
        "{brand} Air Max Running Shoes", "{brand} Ultraboost Sports Shoes", "{brand} Casual Sneakers",
        "{brand} Leather Formal Shoes", "{brand} Flip Flops Comfort", "{brand} Trail Running Shoes",
        "{brand} High-Top Basketball Shoes", "{brand} Walking Shoes Lightweight",
        "{brand} Sandals Open Toe", "{brand} Training Shoes Cross-Fit",
    ],
    "Books": [
        "Introduction to Machine Learning by {brand}", "Data Structures and Algorithms by {brand}",
        "The Art of Programming by {brand}", "Python for Data Science by {brand}",
        "Business Strategy Essentials by {brand}", "Creative Writing Masterclass by {brand}",
        "World History: A Complete Guide by {brand}", "Financial Planning Made Easy by {brand}",
        "Organic Chemistry Fundamentals by {brand}", "Modern Physics Concepts by {brand}",
    ],
    "Sports": [
        "{brand} Badminton Racket Pro", "{brand} Cricket Bat English Willow",
        "{brand} Football Official Size 5", "{brand} Yoga Mat 6mm Premium",
        "{brand} Resistance Bands Set", "{brand} Dumbbells 10kg Pair",
        "{brand} Tennis Racket Graphite", "{brand} Swimming Goggles Anti-Fog",
        "{brand} Boxing Gloves 12oz", "{brand} Skipping Rope Speed",
    ],
    "Home": [
        "{brand} Mixer Grinder 750W", "{brand} Air Purifier HEPA Filter",
        "{brand} Ceiling Fan Energy Efficient", "{brand} LED Bulb 12W Pack of 4",
        "{brand} Iron Steam Press 1600W", "{brand} Water Purifier RO+UV",
        "{brand} Induction Cooktop 2000W", "{brand} Vacuum Cleaner Cordless",
        "{brand} Room Heater 2000W", "{brand} Electric Kettle 1.5L",
    ],
    "Beauty": [
        "{brand} Foundation Matte Finish", "{brand} Moisturizing Face Cream SPF 30",
        "{brand} Hair Serum Anti-Frizz", "{brand} Lipstick Matte Collection",
        "{brand} Sunscreen Lotion SPF 50", "{brand} Face Wash Gentle Cleansing",
        "{brand} Eye Liner Waterproof", "{brand} Body Lotion Cocoa Butter",
        "{brand} Shampoo Anti-Dandruff", "{brand} Perfume Eau de Toilette 100ml",
    ],
    "Toys": [
        "{brand} Building Blocks 500 Pieces", "{brand} Remote Control Car Off-Road",
        "{brand} Board Game Strategy", "{brand} Dollhouse Playset",
        "{brand} Science Kit for Kids", "{brand} Puzzle 1000 Pieces",
        "{brand} Action Figure Collectible", "{brand} Art & Craft Kit",
        "{brand} Musical Instrument Set", "{brand} Outdoor Play Set",
    ],
}

ADJECTIVES = ["Pro", "Ultra", "Lite", "Max", "Plus", "Elite", "Essential", "Advanced"]

COLORS = ["Black", "White", "Blue", "Red", "Grey", "Silver", "Gold", "Navy", "Green", "Pink"]
SIZES = ["S", "M", "L", "XL", "XXL", "Free Size", "28", "30", "32", "34", "36", "7", "8", "9", "10", "11"]
MATERIALS = ["Cotton", "Polyester", "Leather", "Nylon", "Rubber", "Metal", "Plastic", "Wood", "Silicone", "Ceramic"]
WARRANTIES = ["6 months", "1 year", "2 years", "3 years", "No warranty"]


def generate_description(title: str, category: str) -> str:
    """Generate a 2-3 sentence product description (50-80 words)."""
    templates = [
        (
            f"The {title} is designed for everyday use with premium quality materials. "
            f"It features a modern design that combines functionality with style. "
            f"Perfect for those who want reliable {category.lower()} products at a great value."
        ),
        (
            f"Discover the {title}, crafted with attention to detail and built to last. "
            f"Whether you're at home or on the go, this product delivers exceptional performance. "
            f"A top choice in the {category.lower()} category for quality-conscious buyers."
        ),
        (
            f"Introducing the {title} — a perfect blend of innovation and affordability. "
            f"Engineered with cutting-edge technology to meet your daily needs. "
            f"Customers love it for its durability and outstanding value for money."
        ),
        (
            f"Upgrade your lifestyle with the {title}. Built with high-grade components "
            f"and designed for maximum comfort and convenience. "
            f"This {category.lower()} essential is a must-have for every household."
        ),
        (
            f"The {title} stands out for its exceptional build quality and user-friendly design. "
            f"Ideal for daily use, it offers great performance at an unbeatable price. "
            f"A highly rated product in the {category.lower()} segment."
        ),
    ]
    return random.choice(templates)


def generate_attributes(category: str) -> str:
    """Generate a JSON string of product attributes."""
    attrs = {
        "color": random.choice(COLORS),
        "material": random.choice(MATERIALS),
        "warranty": random.choice(WARRANTIES),
    }
    if category in ("Clothing", "Footwear"):
        attrs["size"] = random.choice(SIZES)
    if category == "Electronics":
        attrs["battery"] = random.choice(["Yes", "No"])
        attrs["connectivity"] = random.choice(["Bluetooth", "WiFi", "USB-C", "Wired"])
    return json.dumps(attrs)


def generate_products(num_products: int = 1000) -> list[dict]:
    """Generate a list of product dicts."""
    products = []
    for i in range(1, num_products + 1):
        category = random.choice(CATEGORIES)
        brand = random.choice(BRANDS_BY_CATEGORY[category])
        template = random.choice(PRODUCT_TEMPLATES[category])
        title = template.format(brand=brand, adj=random.choice(ADJECTIVES))
        price = round(random.uniform(199, 49999), 2)
        description = generate_description(title, category)
        attributes = generate_attributes(category)

        products.append({
            "product_id": f"P{i:04d}",
            "title": title,
            "description": description,
            "category": category,
            "price": price,
            "brand": brand,
            "attributes": attributes,
        })
    return products


def generate_user_interactions(products: list[dict], num_interactions: int = 5000) -> list[dict]:
    """Generate simulated user interaction data."""
    product_ids = [p["product_id"] for p in products]
    interaction_types = ["view", "click", "purchase"]
    interaction_weights = [0.70, 0.20, 0.10]
    now = datetime.now()

    interactions = []
    for _ in range(num_interactions):
        user_id = f"U{random.randint(1, 200):03d}"
        product_id = random.choice(product_ids)
        interaction_type = random.choices(interaction_types, weights=interaction_weights, k=1)[0]
        timestamp = now - timedelta(days=random.uniform(0, 90))

        interactions.append({
            "user_id": user_id,
            "product_id": product_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return interactions


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(data_dir, exist_ok=True)

    # Generate products
    print("Generating 5000 products...")
    products = generate_products(5000)
    products_path = os.path.join(data_dir, "products.csv")
    with open(products_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)
    print(f"  Saved {len(products)} products to {products_path}")

    # Generate user interactions
    print("Generating 5000 user interactions...")
    interactions = generate_user_interactions(products, 5000)
    interactions_path = os.path.join(data_dir, "user_interactions.csv")
    with open(interactions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=interactions[0].keys())
        writer.writeheader()
        writer.writerows(interactions)
    print(f"  Saved {len(interactions)} interactions to {interactions_path}")

    print("Done! Data generation complete.")


if __name__ == "__main__":
    main()
